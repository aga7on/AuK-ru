"""Pick extra natural voices (2nd male, 2nd female) from the analyzed corpus for the voices pack."""
import json
import os
import shutil

AUK = r"G:\AI\AuK"
MP = os.path.join(AUK, "local_train", "data", "metrics.jsonl")
CP = os.path.join(AUK, "local_train", "data", "candidates.jsonl")
EXCLUDE = {"b9d5ac1e674ce106.wav"}  # current female ref

text_by_path = {}
for line in open(CP, encoding="utf-8"):
    r = json.loads(line)
    text_by_path[r["path"]] = r["text"]

rows = []
for line in open(MP, encoding="utf-8"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not r.get("ok"):
        continue
    if not (5.5 <= r.get("dur", 0) <= 9.0):
        continue
    if r.get("ovrl", 0) < 3.4 or r.get("silence", 1) > 0.25 or r.get("clipped", 1) > 0.005:
        continue
    if os.path.basename(r["path"]) in EXCLUDE or not os.path.exists(r["path"]):
        continue
    text = text_by_path.get(r["path"], "")
    if len(text) < 40:
        continue
    rows.append((r, text))

males = sorted([x for x in rows if 90 <= x[0].get("f0", 0) <= 150], key=lambda x: -x[0]["ovrl"])
females = sorted([x for x in rows if 215 <= x[0].get("f0", 0) <= 265], key=lambda x: -x[0]["ovrl"])

if males:
    r, t = males[0]
    dst = os.path.join(AUK, "local_tests", "ref_male2.wav")
    shutil.copy(r["path"], dst)
    print("male2: ovrl=%.2f f0=%.0f dur=%.1f -> %s | %s" % (r["ovrl"], r["f0"], r["dur"], dst, t[:50]))
if females:
    r, t = females[0]
    dst = os.path.join(AUK, "local_tests", "ref_female2.wav")
    shutil.copy(r["path"], dst)
    print("female2: ovrl=%.2f f0=%.0f dur=%.1f -> %s | %s" % (r["ovrl"], r["f0"], r["dur"], dst, t[:50]))
