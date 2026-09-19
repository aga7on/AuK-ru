"""Compact report for a variation set results dir (default: local_tests/variations)."""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else r"G:\AI\AuK\local_tests\variations"
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
res = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
by_gen = {r["gen"]: r for r in res}

lines = ["idx | voice | overall | accent | palat | nat | text | wer | issues | text(short)"]
agg = {"male": [], "female": []}
for m in man:
    r = by_gen.get(m["gen"], {})
    j = r.get("judge") or {}
    w = r.get("wer")
    vals = [float(j.get("overall", 0)), float(j.get("accent", 0)),
            float(j.get("palatalization", 0)), float(j.get("naturalness", 0))]
    agg.setdefault(m["voice"], []).append(vals)
    lines.append("%02d | %-6s | %4.0f | %4.0f | %4.0f | %4.0f | %s | wer=%s | %s | %s" % (
        m["idx"], m["voice"], vals[0], vals[1], vals[2], vals[3],
        "Y" if j.get("text_matches") else "N",
        ("%.2f" % w) if isinstance(w, (int, float)) else "?",
        (j.get("issues") or "")[:55], m["text"][:40]))
for v, rows in agg.items():
    if rows:
        n = len(rows)
        mean = [sum(r[i] for r in rows) / n for i in range(4)]
        lines.append("MEAN %-6s overall=%.1f accent=%.1f palat=%.1f nat=%.1f (n=%d)" % (v, mean[0], mean[1], mean[2], mean[3], n))
open(os.path.join(D, "report.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
