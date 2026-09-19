"""Rank a variations results dir: best/worst by judge overall + ASR recall."""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else r"G:\AI\AuK\local_tests\variations_v2"
RES = os.path.join(D, "results_fixed.json")
if not os.path.isfile(RES):
    RES = os.path.join(D, "results.json")
man = {m["gen"]: m for m in json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))}
res = json.load(open(RES, encoding="utf-8"))

rows = []
for r in res:
    m = man.get(r["gen"], {})
    j = r.get("judge") or {}
    rows.append({
        "file": os.path.basename(r["gen"]),
        "idx": m.get("idx"),
        "seed": m.get("seed"),
        "voice": m.get("voice"),
        "overall": j.get("overall", 0),
        "nat": j.get("naturalness", 0),
        "prosody": j.get("prosody", 0),
        "stress": j.get("stress", 0),
        "endings": j.get("endings", 0),
        "recall": r.get("recall", 0),
        "missing": r.get("missing_words", []),
        "text": (m.get("text") or "")[:50],
    })

by_quality = sorted(rows, key=lambda x: (x["recall"], x["overall"], x["nat"]))
print("=== WORST 8 (по recall, потом overall) ===")
for x in by_quality[:8]:
    print("  %-22s ovrl=%-3s nat=%-3s recall=%.2f missing=%s | %s" % (
        x["file"], x["overall"], x["nat"], x["recall"], str(x["missing"])[:40], x["text"]))
print("=== BEST 8 ===")
for x in by_quality[-8:][::-1]:
    print("  %-22s ovrl=%-3s nat=%-3s prosody=%-3s recall=%.2f | %s" % (
        x["file"], x["overall"], x["nat"], x["prosody"], x["recall"], x["text"]))

n = len(rows)
mean = lambda k: sum(x[k] for x in rows) / max(n, 1)
print("MEAN: overall=%.1f nat=%.1f prosody=%.1f stress=%.1f endings=%.1f recall=%.2f (n=%d)" % (
    mean("overall"), mean("nat"), mean("prosody"), mean("stress"), mean("endings"), mean("recall"), n))


def per_voice(v):
    sel = [x for x in rows if x["voice"] == v]
    if not sel:
        return
    print("MEAN %s: overall=%.1f recall=%.2f (n=%d)" % (
        v, sum(x["overall"] for x in sel) / len(sel), sum(x["recall"] for x in sel) / len(sel), len(sel)))


for v in ("male", "female"):
    per_voice(v)
