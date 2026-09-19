"""Aggregate autopsy results: per-category stats + hotlist for human review."""
import json
import os
from collections import defaultdict

D = r"G:\AI\AuK\local_tests\autopsy"
res = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
meta = {m["gen"].lower(): m for m in man}

rows = []
for e in res:
    m = meta.get(e["gen"].lower(), {})
    j = e.get("judge") or {}
    o = e.get("objective") or {}
    rows.append({
        "idx": e["idx"],
        "cat": m.get("category", "?"),
        "seed": m.get("seed"),
        "file": os.path.basename(e["gen"]),
        "overall": j.get("overall"),
        "fidelity": j.get("text_fidelity"),
        "prosody": j.get("prosody"),
        "nat": j.get("naturalness"),
        "gender_ok": j.get("same_gender"),
        "trunc": j.get("truncated"),
        "misstressed": j.get("words_misstressed") or [],
        "issues": (j.get("issues") or "")[:90],
        "dur_ratio": o.get("dur_ratio"),
        "ref": os.path.basename(m.get("ref", "")),
        "text": e.get("text", "")[:70],
    })

by_cat = defaultdict(list)
for r in rows:
    by_cat[r["cat"]].append(r)

print("=== PER-CATEGORY (local judge) ===")
for cat, rs in sorted(by_cat.items()):
    ov = [r["overall"] for r in rs if r["overall"] is not None]
    fid = [r["fidelity"] for r in rs if r["fidelity"] is not None]
    pr = [r["prosody"] for r in rs if r["prosody"] is not None]
    gbad = sum(1 for r in rs if r["gender_ok"] is False)
    tr = sum(1 for r in rs if r["trunc"])
    mis = sum(1 for r in rs if r["misstressed"])
    mn = min(ov) if ov else "?"
    print(f"{cat:<11} n={len(rs):<3} overall mean={sum(ov)/len(ov):.1f} min={mn} | fid={sum(fid)/len(fid):.1f} pros={sum(pr)/len(pr):.1f} | gender_bad={gbad} trunc={tr} misstressed={mis}")

print()
print("=== HOTLIST (gender flip / trunc / overall<=7 / misstress) ===")
hot = [r for r in rows if (r["gender_ok"] is False) or r["trunc"] or (r["overall"] is not None and r["overall"] <= 7) or r["misstressed"]]
for r in sorted(hot, key=lambda x: (x["overall"] if x["overall"] is not None else 0)):
    flags = []
    if r["gender_ok"] is False:
        flags.append("GENDER")
    if r["trunc"]:
        flags.append("TRUNC")
    if r["misstressed"]:
        flags.append("MISSTRESS:" + ",".join(r["misstressed"][:3]))
    print(f"{r['file']:<34} ov={r['overall']} {';'.join(flags)} | {r['issues']}")

print()
print("=== MALE vs FEMALE REF ===")
for refname in ("user_ref.wav", "ref_ru.wav"):
    rs = [r for r in rows if r["ref"] == refname]
    ov = [r["overall"] for r in rs if r["overall"] is not None]
    gbad = sum(1 for r in rs if r["gender_ok"] is False)
    print(f"{refname}: n={len(rs)} overall mean={sum(ov)/len(ov):.1f} min={min(ov)} gender_bad={gbad}")

json.dump({"rows": rows}, open(os.path.join(D, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved summary.json")
