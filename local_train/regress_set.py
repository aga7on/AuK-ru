"""Overnight release regression on u18000 through the product path (run_generate).

1) 45 phrases (autopsy set) x best-of-2 + trim -> regress_u18000/
2) flaky pack: 'Учитель физики...' x 6 seeds (no best-of) -> quantifies stochastic tails
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

import auk.infer.infer_gradio as ig  # noqa: E402
from auk.infer.infer_auk import AukInfer  # noqa: E402
from auk.infer import quality  # noqa: E402
from autopsy_set import PHRASES  # noqa: E402

REF = os.path.join(AUK, "local_tests", "user_ref.wav")
MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
OUT = os.path.join(AUK, "local_tests", "regress_u18000")

ig.CKPT_PATHS["rg"] = MERGED
ig.ENGINES["rg"] = AukInfer(
    config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
    ckpt_path=MERGED,
    qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
    cpu_offload=True,
    device="cuda:1",
    dtype="bf16",
)


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for idx, (text, cat) in enumerate(PHRASES):
        sr, pcm = ig.run_generate("rg", REF, text, None, "", "", 64, 2.0, 1234,
                                  translit=False, accentize=True, bestofn=2, trim=True, tts_wrap=True)
        path = os.path.join(OUT, f"rg{idx:02d}_{cat}.wav")
        import soundfile as sf
        sf.write(path, pcm, sr, subtype="PCM_16")
        import torch
        heard = quality.transcribe(torch.from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0), sr)
        rec = quality.recall(text, heard)
        rows.append({"idx": idx, "cat": cat, "text": text, "file": path,
                     "recall": round(rec, 3), "asr": heard})
        flag = "OK " if rec >= 0.9 else "!! "
        print(f"{flag}rg{idx:02d} {cat}: recall={rec:.2f}", flush=True)

    # flaky pack: 6 plain seeds of the fragile phrase
    fragile = "Учитель физики объяснял законы сохранения энергии."
    for si, seed in enumerate([1234, 1235, 7, 42, 99, 2024]):
        sr, pcm = ig.ENGINES["rg"].generate(
            [{"role": "user", "content": [
                {"type": "text", "text": f"Say the following in Russian with clear, natural pronunciation: '{ig._accentize_ru(fragile)}'"},
                {"type": "audio", "audio": REF},
            ]}],
            audio=REF, gen_seconds=estimate_seconds(fragile), nfe=64, cfg_strength=2.0, seed=seed)
        pcm = quality.trim_silence(pcm, sr)
        path = os.path.join(OUT, f"flaky_s{seed}.wav")
        import soundfile as sf
        sf.write(path, pcm, sr, subtype="PCM_16")
        import torch
        heard = quality.transcribe(torch.from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0), sr)
        rec = quality.recall(fragile, heard)
        rows.append({"idx": f"flaky", "cat": f"seed{seed}", "text": fragile, "file": path,
                     "recall": round(rec, 3), "asr": heard})
        print(f"flaky s{seed}: recall={rec:.2f}", flush=True)

    json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in rows if r["recall"] >= 0.95)
    print(f"REGRESS_DONE: {ok}/{len(rows)} files with recall>=0.95")


if __name__ == "__main__":
    main()
