"""Inspect ASR/recall for suspect autopsy files (no GPU needed)."""
import json
import os
import sys

d = json.load(open(r"G:\AI\AuK\local_tests\autopsy\results.json", encoding="utf-8"))
want = sys.argv[1:] or [
    "ap00_hard_words_s7",
    "ap02_hard_words_s7",
    "ap05_reduction_s1234",
    "ap19_soft_sign_s1234",
    "ap30_baseline_s1234",
    "ap41_baseline_s1234",
]
ZDOR = "\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435"
ROL = "\u0440\u043e\u043b\u044c"
for e in d:
    f = os.path.basename(e["gen"])
    if f.startswith(tuple(want)):
        asr = e.get("asr") or ""
        print(f"{f}: recall={e.get('recall')} missing={e.get('missing_words')} "
              f"has_zdor={ZDOR in asr.lower()} has_rol={ROL in asr.lower()}")
        print("   asr :", asr.encode("ascii", "replace").decode()[:120])
        print("   text:", e["text"].encode("ascii", "replace").decode()[:90])
