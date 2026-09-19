"""Scan instruction formats in train/val data + check UI wiring facts."""
import json
import re
import os
from collections import Counter

AUK = r"G:\AI\AuK"
pref = Counter()
n = 0
for fname in ("train.jsonl", "val.jsonl"):
    path = os.path.join(AUK, "local_train", "data", fname)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 6000:
                break
            row = json.loads(line)
            for m in row.get("messages", []):
                if m.get("role") == "user":
                    for c in (m.get("content") or []):
                        if c.get("type") == "text":
                            t = c["text"]
                            m0 = re.match(r"(.{0,60}?)[\"'\u00ab\u00bb]", t)
                            pref[(fname, m0.group(1) if m0 else t[:60])] += 1
                            n += 1
for (fn, p), cnt in pref.most_common(15):
    print(f"{fn:11} {cnt:6} | {p.encode('ascii','backslashreplace').decode()}")

print()
for p in ("ckpts/AuK-Flash/auk_flash.safetensors", "ckpts/AuK/auk_base.safetensors"):
    print(p, os.path.exists(os.path.join(AUK, p)))
