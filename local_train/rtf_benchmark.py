# -*- coding: utf-8 -*-
"""S14: RTF-бенчмарк инференса v1.0 (real-time factor = wall_time / audio_duration).

Матрица: nfe (64/32) × режим (TTS без рефа / clone с рефом) × 3 повтора.
Дополнительно: время загрузки модели и время composite-rerank одного кандидата
(стоимость best-of-3 = 3×gen + 3×rerank).

usage: python rtf_benchmark.py [--device cuda:0]
Выход: local_train/reports/deepseek_supervised/S14_RTF.md + stdout.
"""
import argparse
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "S14_RTF.md")

TEXTS = [
    "Привет! Это проверка скорости синтеза русской речи.",
    "Сегодня отличный день, и мы тестируем новую систему синтеза с замером реального времени.",
]
REF = r"G:\AI\kyutai-ru\data\ru_wav_mfa\45cc9d96e6beeb10.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak

    t0 = time.time()
    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"),
                   ckpt_path=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")
    load_s = time.time() - t0
    print(f"model load: {load_s:.1f}s")

    outdir = os.path.join(AUK, "local_tests", "rtf_bench")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for nfe in (64, 32):
        for mode in ("tts", "clone"):
            for ti, text in enumerate(TEXTS[:1]):
                for rep in range(args.reps):
                    instr = (f"Say the following in Russian with clear, natural pronunciation: '{text}'"
                             if mode == "tts" else
                             f"Reproduce the reference voice and say in Russian: '{text}'")
                    content = [{"type": "text", "text": instr}]
                    kw = {}
                    if mode == "clone":
                        content.append({"type": "audio", "audio": REF})
                        kw["audio"] = REF
                        gs = float(sf.info(REF).duration) + 0.5
                    else:
                        gs = 6.0
                    t = time.time()
                    audio, sr = eng.generate([{"role": "user", "content": content}],
                                             gen_seconds=gs, nfe=nfe, cfg_strength=2.0, seed=7, **kw)
                    gen_s = time.time() - t
                    audio = limit_peak(normalize_rms(audio))
                    a1 = audio.mean(dim=0) if getattr(audio, "ndim", 1) > 1 else audio
                    dur = float(a1.shape[-1]) / sr
                    fp = os.path.join(outdir, f"nfe{nfe}_{mode}_r{rep}.wav")
                    save_audio(audio, sr, fp)
                    rows.append({"nfe": nfe, "mode": mode, "rep": rep,
                                 "gen_s": round(gen_s, 2), "audio_s": round(dur, 2),
                                 "rtf": round(gen_s / dur, 3)})
                    print(f"nfe={nfe} {mode} rep{rep}: gen {gen_s:.1f}s audio {dur:.1f}s RTF {gen_s/dur:.3f}", flush=True)

    # rerank cost (1 кандидат)
    t = time.time()
    from rerank_composite import naturalness_score, loudness_score, pause_score, repetition_penalty, artifact_penalty
    import numpy as np
    x, sr = sf.read(os.path.join(outdir, "nfe64_tts_r0.wav"), dtype="float32")
    _ = naturalness_score(x, sr); _ = loudness_score(x); _ = pause_score(x, sr)
    _ = repetition_penalty("тест тест"); _ = artifact_penalty(x)
    rerank_dsp_s = time.time() - t
    t = time.time()
    from ru_metrics import transcribe_path
    _, _ = transcribe_path(os.path.join(outdir, "nfe64_tts_r0.wav"))
    asr_s = time.time() - t
    t = time.time()
    from speaker_embed import embed_path
    _ = embed_path(os.path.join(outdir, "nfe64_tts_r0.wav"))
    emb_s = time.time() - t

    import statistics as st
    L = ["# S14 — RTF-бенчмарк v1.0 (s7@5750)", "",
         f"Устройство: {args.device}, cpu_offload=True, bf16. Загрузка модели: {load_s:.0f} c.", "",
         "| nfe | режим | RTF median | gen s | audio s |", "|---|---|---|---|---|"]
    for nfe in (64, 32):
        for mode in ("tts", "clone"):
            rs = [r for r in rows if r["nfe"] == nfe and r["mode"] == mode]
            if rs:
                L.append(f"| {nfe} | {mode} | {st.median(r['rtf'] for r in rs):.3f} | "
                         f"{st.median(r['gen_s'] for r in rs):.1f} | {st.median(r['audio_s'] for r in rs):.1f} |")
    L += ["", "## Стоимость reranker-компонентов (на 1 кандидата)", "",
          f"- DSP-метрики (dnsmos+loud+pause+penalties): {rerank_dsp_s:.2f} c",
          f"- GigaAM ASR: {asr_s:.2f} c",
          f"- WeSpeaker embedding: {emb_s:.2f} c",
          f"- итого best-of-3 overhead: ≈ {3*(rerank_dsp_s+asr_s+emb_s):.1f} c на запрос", "",
          "## Вывод", "",
          "RTF < 1 = быстрее реального времени. Цель ROADMAP S14: RTF < 1, затем 0.3–0.5.",
          "Следующие рычаги (не испробованы): nfe-дистилляция (AuK-Flash 4-step), torch.compile,",
          "батчинг кандидатов, отключение cpu_offload при VRAM ≥ 24 ГБ."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print(f"rerank overhead per candidate: dsp {rerank_dsp_s:.2f}s asr {asr_s:.2f}s emb {emb_s:.2f}s")


if __name__ == "__main__":
    main()
