# -*- coding: utf-8 -*-
"""Микро-эксперимент S9.2: какая форма записи аббревиатур произносима моделью.

Варианты: raw (МГУ), spaces (эм гэ у), hyphens (эм-гэ-у), dots (э.м.г.у.).
2 текста × 4 формы × 2 сида = 16 генераций. Метрика: GigaAM WER против expected.

usage: python abbr_form_probe.py [--device cuda:1]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio
from auk.infer.ru_frontend import to_speakable
from ru_metrics import transcribe_path, text_metrics

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "abbr_form_probe")

FORMS = {
    "raw": "МГУ им. Ломоносова объявил набор.",
    "spaces": "эм гэ у им. Ломоносова объявил набор.",
    "hyphens": "эм-гэ-у им. Ломоносова объявил набор.",
    "dots": "э.м.г.у. им. Ломоносова объявил набор.",
    "raw2": "ООО Ромашка подписало договор.",
    "spaces2": "о о о Ромашка подписало договор.",
    "hyphens2": "о-о-о Ромашка подписало договор.",
    "dots2": "о.о.о. Ромашка подписало договор.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    ref = next(e["ref"] for e in pack if e.get("kind") == "clone" and e["id"] == "clone08")

    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"),
                   ckpt_path=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    rows = []
    t0 = time.time()
    for form, raw in FORMS.items():
        for seed in (7, 999):
            instr = f"Say the following in Russian with clear, natural pronunciation: '{raw}'"
            fp = os.path.join(OUT, f"{form}_s{seed}.wav")
            audio, sr = eng.generate([{"role": "user", "content": [{"type": "text", "text": instr}]}],
                                     audio=ref, gen_seconds=6.0, nfe=64, cfg_strength=2.0, seed=seed)
            from auk.infer.quality import normalize_rms, limit_peak
            audio = limit_peak(normalize_rms(audio))
            save_audio(audio, sr, fp)
            heard, err = transcribe_path(fp)
            tm = text_metrics(raw.replace("+", ""), heard)
            rows.append({"form": form, "seed": seed, "wer": round(tm["wer"], 3),
                         "heard": (heard or "")[:90]})
            print(f"{form} s{seed} wer={tm['wer']:.3f} heard={(heard or '')[:60]}", flush=True)

    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("--- by form ---")
    by = {}
    for r in rows:
        by.setdefault(r["form"], []).append(r["wer"])
    for f, ws in sorted(by.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"{f}: mean_wer={sum(ws)/len(ws):.3f} {ws}")
    print(f"ABBR_FORM_DONE {len(rows)} in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
