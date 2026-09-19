"""Pick natural female reference candidates from the analyzed corpus (metrics.jsonl)."""
import json
import os

MP = r"G:\AI\AuK\local_train\data\metrics.jsonl"
CP = r"G:\AI\AuK\local_train\data\candidates.jsonl"

text_by_path = {}
for line in open(CP, encoding="utf-8"):
    r = json.loads(line)
    text_by_path[r["path"]] = r["text"]

cands = []
for line in open(MP, encoding="utf-8"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not r.get("ok"):
        continue
    if not (5.5 <= r.get("dur", 0) <= 9.0):
        continue
    if not (185 <= r.get("f0", 0) <= 270):
        continue
    if r.get("ovrl", 0) < 3.3 or r.get("sig", 0) < 3.3:
        continue
    if r.get("silence", 1) > 0.25 or r.get("clipped", 1) > 0.005 or r.get("peak", 0) > 0.97:
        continue
    text = text_by_path.get(r["path"], "")
    if len(text) < 40:
        continue
    if not os.path.exists(r["path"]):
        continue
    cands.append((r, text))

cands.sort(key=lambda x: -x[0]["ovrl"])
print(f"candidates: {len(cands)}")
for r, t in cands[:12]:
    print("ovrl=%.2f sig=%.2f f0=%.0f dur=%.1f sil=%.2f | %s | %s" % (
        r["ovrl"], r["sig"], r["f0"], r["dur"], r["silence"], r["path"], t[:60]))

with open(r"G:\AI\AuK\local_train\female_ref_candidates.json", "w", encoding="utf-8") as f:
    json.dump([{"path": r["path"], "text": t, "metrics": {k: r[k] for k in ("ovrl", "sig", "f0", "dur", "silence")}}
               for r, t in cands[:8]], f, ensure_ascii=False, indent=1)
print("saved top-8 to female_ref_candidates.json")
