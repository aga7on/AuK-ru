"""WER у best-of-3 (по sim) выходов seed-проба: клон-гейт WER ≤0.15 на выбранных."""
import json
import os
import re
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train")

from ru_metrics import transcribe_path, text_metrics

probe = json.load(open(r"G:\AI\AuK\local_tests\tmp_seed_probe\seed_probe.json", encoding="utf-8"))
pack = json.load(open(r"G:\AI\AuK\local_tests\eval_pack\pack.json", encoding="utf-8"))
texts = {}
for e in pack:
    if e.get("kind") == "clone":
        m = re.search(r"'([^']*)'", e.get("instruction", ""))
        texts[e["id"]] = m.group(1) if m else ""

rows = []
for cid, sims in probe["per_clone"]:
    best_seed = (7, 123, 999)[sims.index(max(sims))]
    p = rf"G:\AI\AuK\local_tests\tmp_seed_probe\{cid}_s{best_seed}.wav"
    heard, err = transcribe_path(p)
    tm = text_metrics(texts[cid], heard)
    rows.append({"id": cid, "seed": best_seed, "sim": max(sims), "wer": round(tm["wer"], 3)})
    print(cid, "seed", best_seed, "sim", max(sims), "wer", round(tm["wer"], 3), flush=True)

import statistics as st
wers = [r["wer"] for r in rows]
print("WER median", st.median(wers), "mean", round(st.mean(wers), 3))
json.dump(rows, open(r"G:\AI\AuK\local_tests\tmp_seed_probe\best_wer.json", "w", encoding="utf-8"), indent=1)
