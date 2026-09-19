"""Объективные замеры для 100-клонового стресс-теста s5: WeSpeaker sim + GigaAM WER.
Гейт: sim median >= 0.75, WER median <= 0.20, доля sim < 0.6, доля WER > 0.5.
Пишет local_tests/s5_clone100/objective.json.
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "local_train"))

from ru_metrics import transcribe_path, text_metrics
from speaker_embed import embed_path

import numpy as np

OUT = os.path.join(AUK, "local_tests", "s5_clone100")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT
    results = json.load(open(os.path.join(out_dir, "results.json"), encoding="utf-8"))
    rows = []
    for i, r in enumerate(results):
        if r["status"] != "ok":
            rows.append({"id": r["id"], "status": r["status"]})
            continue
        wav, ref = r["file"], r["ref"]
        heard, asr_err = transcribe_path(wav)
        tm = text_metrics(r["text"], heard)
        try:
            sim = round(cosine(embed_path(wav), embed_path(ref)), 4)
        except Exception as e:
            sim = None
        rows.append({"id": r["id"], "wer": round(tm["wer"], 3), "sim": sim,
                     "asr_error": asr_err, "heard": heard,
                     "subs": tm.get("subs", [])})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(results)}]", flush=True)

    ok = [r for r in rows if r.get("sim") is not None]
    sims = [r["sim"] for r in ok]
    wers = [r["wer"] for r in ok]
    agg = {
        "n": len(ok),
        "sim_median": round(float(np.median(sims)), 4),
        "sim_mean": round(float(np.mean(sims)), 4),
        "sim_lt_0.6": sum(1 for s in sims if s < 0.6),
        "sim_lt_0.65": sum(1 for s in sims if s < 0.65),
        "wer_median": round(float(np.median(wers)), 3),
        "wer_mean": round(float(np.mean(wers)), 3),
        "wer_gt_0.5": sum(1 for w in wers if w > 0.5),
        "both_ok (sim>=0.6 and wer<=0.3)": sum(1 for r in ok if r["sim"] >= 0.6 and r["wer"] <= 0.3),
    }
    json.dump({"aggregate": agg, "rows": rows},
              open(os.path.join(out_dir, "objective.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
