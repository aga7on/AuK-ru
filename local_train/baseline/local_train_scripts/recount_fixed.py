"""Recompute recall/missing for stored eval results using the fixed normalizer (no GPU)."""
import json
import os
import sys
from collections import Counter

LOCAL = r"G:\AI\AuK\local_train"
sys.path.insert(0, LOCAL)
sys.path.insert(0, r"G:\AI\AuK\src")

from judge_eval import recall_and_missing  # noqa: E402

DIRS = [
    r"G:\AI\AuK\local_tests\variations_v2",
    r"G:\AI\AuK\local_tests\variations",
]


def process(d):
    rp = os.path.join(d, "results.json")
    if not os.path.isfile(rp):
        print(f"{d}: no results.json")
        return
    res = json.load(open(rp, encoding="utf-8"))
    for r in res:
        recall, missing = recall_and_missing(r.get("text", ""), r.get("asr", ""))
        r["recall_fixed"] = round(recall, 3)
        r["missing_fixed"] = missing
        r["wer"] = round(1.0 - recall, 3)
    out = os.path.join(d, "results_fixed.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    man = {}
    mp = os.path.join(d, "manifest.json")
    if os.path.isfile(mp):
        man = {m["gen"]: m for m in json.load(open(mp, encoding="utf-8"))}

    rows = []
    for r in res:
        m = man.get(r["gen"], {})
        j = r.get("judge") or {}
        rows.append({
            "file": os.path.basename(r["gen"]),
            "voice": m.get("voice", "?"),
            "overall": j.get("overall", 0),
            "nat": j.get("naturalness", 0),
            "recall": r["recall_fixed"],
            "missing": r["missing_fixed"],
            "text": (m.get("text") or "")[:45],
        })
    rows.sort(key=lambda x: (x["recall"], -x["overall"]))
    print(f"== {os.path.basename(d)} ({len(rows)} files)")
    print("  worst 6:")
    for x in rows[:6]:
        print("   %-24s ovrl=%-3s recall=%.2f missing=%s | %s" % (
            x["file"], x["overall"], x["recall"], str(x["missing"])[:50], x["text"]))
    n = len(rows)
    print("  MEAN recall=%.2f overall=%.1f" % (
        sum(x["recall"] for x in rows) / max(n, 1), sum(x["overall"] for x in rows) / max(n, 1)))
    for v in ("male", "female"):
        sel = [x for x in rows if x["voice"] == v]
        if sel:
            print("  MEAN %-6s recall=%.2f overall=%.1f (n=%d)" % (
                v, sum(x["recall"] for x in sel) / len(sel), sum(x["overall"] for x in sel) / len(sel), len(sel)))


for d in DIRS:
    process(d)
