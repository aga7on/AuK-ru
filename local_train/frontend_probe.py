# -*- coding: utf-8 -*-
"""S9 probe: frontend → TTS → ASR-WER на hard_cases_ru.jsonl (v1.0 = s7@5750).

Метод: 30 текстов (3 на категорию, seed 7), инструкция формата eval-pack TTS
(«Say the following in Russian with clear, natural pronunciation: '<to_speakable(text)>'»).
Ожидаемый текст для WER — нормализованный (без маркеров ударения).
Гейт пробы: WER mean ≤ 0.20 и first_ok ≥ 0.70 (те же пороги GATES §1 TTS).

usage: python frontend_probe.py [--n_per_cat 3] [--ckpt ...] [--device cuda:1]
Выход: local_tests\frontend_probe_s9\results.json + сводка в stdout.
"""
import argparse
import json
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio
from auk.infer.ru_frontend import to_speakable
from ru_metrics import transcribe_path, text_metrics
from wer_norm import norm_for_wer

AUK = r"G:\AI\AuK"
OUT_DEFAULT = os.path.join(AUK, "local_tests", "frontend_probe_s9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cat", type=int, default=3)
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"))
    ap.add_argument("--config", default=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    bench = [json.loads(l) for l in open(os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru.jsonl"), encoding="utf-8") if l.strip()]
    by_cat = {}
    for r in bench:
        by_cat.setdefault(r["category"], []).append(r)
    sel = []
    for cat, rs in by_cat.items():
        sel.extend(rs[: args.n_per_cat])
    print("selected:", len(sel), "of", len(bench))

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    ref = next(e["ref"] for e in pack if e.get("kind") == "clone" and e["id"] == "clone08")

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    rows = []
    t0 = time.time()
    for i, r in enumerate(sel):
        speak = to_speakable(r["text"])
        expected = speak.replace("+", "")
        instr = f"Say the following in Russian with clear, natural pronunciation: '{speak}'"
        fpath = os.path.join(OUT, f"{r['id']}.wav")
        try:
            audio, sr = eng.generate([{"role": "user", "content": [{"type": "text", "text": instr}]}],
                                     audio=ref, gen_seconds=max(3.5, min(14.0, 1.0 + 0.09 * len(expected))),
                                     nfe=64, cfg_strength=2.0, seed=7)
            from auk.infer.quality import normalize_rms, limit_peak
            audio = limit_peak(normalize_rms(audio))
            save_audio(audio, sr, fpath)
            heard, asr_err = transcribe_path(fpath)
            tm = text_metrics(expected, heard)
            tmn = text_metrics(norm_for_wer(expected), norm_for_wer(heard))
            rows.append({"id": r["id"], "category": r["category"], "raw": r["text"],
                         "speakable": speak, "expected": expected, "heard": heard,
                         "wer": round(tm["wer"], 3), "wer_norm": round(tmn["wer"], 3),
                         "cer_norm": round(tmn.get("cer", -1), 3), "asr_error": asr_err,
                         "file": fpath, "status": "ok"})
        except Exception as e:
            rows.append({"id": r["id"], "category": r["category"], "raw": r["text"],
                         "status": f"error {type(e).__name__}: {e}"})
        print(f"[{i+1}/{len(sel)}] {r['id']} {rows[-1]['status']} wer={rows[-1].get('wer')} ({(time.time()-t0)/60:.1f}m)", flush=True)

    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in rows if r["status"] == "ok"]
    wers = [r.get("wer_norm", r["wer"]) for r in ok]
    first_ok = sum(1 for w in wers if w <= 0.15)
    mean = sum(wers) / len(wers) if wers else None
    print(f"FRONTEND_PROBE_DONE n={len(ok)}/{len(rows)} wer_mean={mean} first_ok={first_ok}/{len(ok)}")
    by = {}
    for r in ok:
        by.setdefault(r["category"], []).append(r.get("wer_norm", r["wer"]))
    for c, ws in sorted(by.items()):
        print(f"  {c}: n={len(ws)} wer_mean={sum(ws)/len(ws):.3f}")


if __name__ == "__main__":
    main()
