"""Dump compact ASR reference table (recall/missing) for autopsy files."""
import json
import os

D = r"G:\AI\AuK\local_tests\autopsy"
d = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
d = sorted(d, key=lambda e: e["idx"])
lines = ["idx | file | recall | missing | asr_first_60"]
for e in d:
    f = os.path.basename(e["gen"])
    missing = ",".join(e.get("missing_words") or [])
    asr = (e.get("asr") or "").split("Human:")[0].strip()[:60]
    lines.append(f"{e['idx']:02d} | {f} | {e.get('recall')} | {missing} | {asr}")
out = os.path.join(D, "asr_table.txt")
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("saved", out, len(lines) - 1, "rows")
low = [l for l in lines[1:] if float(l.split(" | ")[2]) < 1.0]
print("recall<1.0:", len(low), "of", len(lines) - 1)
