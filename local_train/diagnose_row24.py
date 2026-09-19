"""Диагностика four-signals: оригинал WAV → VAE reconstruction → генерация.
Для примера «Сбер ты низкая» (row 24 из s2 train.jsonl).
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_10000.safetensors")
OUT = os.path.join(AUK, "local_train", "diagnostics")
sys.path.insert(0, r"G:\AI\AuK\src")

row = json.load(open(os.path.join(AUK, "local_train", "diagnostics", "problem_row24.json"), encoding="utf-8"))
tgt_path = row["row"]["messages"][1]["content"][0]["audio_url"]
ref_path = row["row"]["messages"][0]["content"][0].get("audio")
text = row["row"]["messages"][0]["content"][0].get("text", "")
print("target:", tgt_path)
print("ref:", ref_path)
print("instruction:", row["row"]["messages"][0]["content"][0]["text"][:120])


def audio_stats(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    return {"dur": round(len(x) / sr, 3), "sr": sr, "peak": round(peak, 3), "rms": round(rms, 4)}


def main():
    from gigaam_asr import transcribe
    from auk.infer.infer_auk import AukInfer, save_audio

    results = {}

    # === signal 1: оригинальный WAV ===
    x, sr = sf.read(tgt_path, dtype="float32", always_2d=True)
    x_mono = x.mean(axis=1)
    results["original"] = audio_stats(x_mono, sr)
    t1 = transcribe(tgt_path)
    results["original_asr"] = t1
    # повтор ASR 3 раза для детерминизма
    t1b = transcribe(tgt_path)
    t1c = transcribe(tgt_path)
    results["original_asr_repeat"] = [t1[:80], t1b[:80], t1c[:80]]
    results["original_deterministic"] = (t1 == t1b == t1c)

    print("\n=== SIGNAL 1: ORIGINAL WAV ===")
    print("stats:", results["original"])
    print("asr:", t1[:80])
    print("deterministic:", results["original_asr_repeat"])

    # === signal 2: после VAE reconstruction (decode+encode) ===
    import torchaudio
    wave = torch.from_numpy(np.ascontiguousarray(x_mono, dtype=np.float32)).unsqueeze(0)
    if sr != 24000:
        wave = torchaudio.functional.resample(wave, sr, 24000)
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
        ckpt_path=MERGED,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:0",
        dtype="bf16",
    )
    vae = engine.vae_model
    # pad to even number of latent frames (hop_size=480 → 960 for even)
    hop = 480
    if wave.shape[-1] // hop % 2 != 0:
        wave = torch.nn.functional.pad(wave, (0, hop))
    # add channel dim: [B, T] -> [B, 1, T]
    wave = wave.unsqueeze(1)
    wave = wave.to(next(vae.parameters()).device)
    print(f"WAVE shape: {wave.shape}, device: {wave.device}, hop: {hop}, frames: {wave.shape[-1]//hop}")
    ref_latents, ref_lens = vae.encoding_and_normalization(wave, sample_lengths=torch.tensor([wave.shape[-1]]).to(wave.device))
    print(f"LATENTS shape: {ref_latents.shape}, lens: {ref_lens}")
    recon = vae.inference_from_latents(ref_latents.permute(0, 2, 1))
    recon_np = recon.detach().cpu().numpy().astype(np.float32)
    if recon_np.ndim == 3:
        recon_np = recon_np.squeeze(0).squeeze(0) if recon_np.shape[0] == 1 else recon_np.reshape(-1)
    results["vae_reconstruction"] = audio_stats(recon_np, 24000)
    recon_path = os.path.join(AUK, "local_train", "diagnostics", "recon_row24.wav")
    import soundfile as sf2
    sf2.write(recon_path, recon_np, 24000)
    t2 = transcribe(recon_path)
    results["vae_recon_asr"] = t2
    print("\n=== SIGNAL 2: VAE RECONSTRUCTION ===")
    print("stats:", results["vae_reconstruction"])
    print("asr:", t2[:80])

    # === signal 3: генерация с base (u0) ===
    print("\n=== SIGNAL 3: GENERATION BASE u10000 (same input) ===")
    print("\n=== SIGNAL 3: GENERATION BASE u10000 (same input) ===")
    from auk.infer.infer_gradio import _accentize_ru
    # правильный путь: взять текст из content[0] и ref из content[1]
    raw_instr = row["row"]["messages"][0]["content"][0]["text"]
    stressed = raw_instr.replace("Reproduce the reference voice and say in Russian: '", "").replace("Say the following in Russian with clear, natural pronunciation: '", "").replace("'", "")
    gen_ref = row["row"]["messages"][0]["content"][1]["audio"] if len(row["row"]["messages"][0]["content"]) > 1 else None
    print("gen_ref:", gen_ref, "exists:", os.path.exists(gen_ref) if gen_ref else False)
    gen_secs = 4.6
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Say the following with the same voice: '{stressed}'"},
        {"type": "audio", "audio": gen_ref},
    ]}]
    gen_audio, gen_sr = engine.generate(messages, audio=gen_ref, gen_seconds=gen_secs,
                                        nfe=64, cfg_strength=2.0, seed=1234)
    gen_path = os.path.join(OUT, "gen_row24_base.wav")
    save_audio(gen_audio, gen_sr, gen_path)
    gen_np = gen_audio.detach().cpu().numpy().astype(np.float32)
    if gen_np.ndim > 1:
        gen_np = gen_np.mean(axis=0) if gen_np.shape[0] <= 8 else gen_np.mean(axis=1)
    results["generation"] = audio_stats(gen_np, gen_sr)
    t3 = transcribe(gen_path)
    results["generation_asr"] = t3
    print("stats:", results["generation"])
    print("asr:", t3[:80])

    # === ref audio signal ===
    if ref_path and os.path.exists(ref_path):
        rx, rsr = sf.read(ref_path, dtype="float32", always_2d=True)
        results["ref_audio"] = audio_stats(rx.mean(axis=1), rsr)
        results["ref_asr"] = transcribe(ref_path)
        print("\n=== REF AUDIO ===")
        print("stats:", results["ref_audio"])
        print("asr:", results["ref_asr"][:80])

    json.dump(results, open(os.path.join(OUT, "four_signals.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nFOUR_SIGNALS_DONE")


if __name__ == "__main__":
    main()
