import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(HERE, "diag_out", "progress_manifest.json"), encoding="utf-8"))
keys = ("control", "fixture", "status", "output", "output_sha256", "duration_s", "sr",
        "peak", "clip_ratio_ge_0.985", "runtime_s")
for i in m["items"]:
    print("|".join(str(i[k]) for k in keys))
print("SUMMARY", json.dumps(m["summary"], ensure_ascii=False))
