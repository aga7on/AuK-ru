"""B-mix: 80/20 речь/tools для пилота B.

Речь: data_s2_full/v2_after_identity/train.jsonl (не изменяется).
Tools: стратифицированная выборка из data_s2_tools_v3/train.jsonl (~20% итогового файла).
Детерминированно (seed 7). Итоговый файл: data_s2_full/B_mix_80_20/train.jsonl
"""
import json
import os
import random
from collections import Counter

LOCAL = r"G:\AI\AuK\local_train"
SPEECH = os.path.join(LOCAL, "data_s2_full", "v2_after_identity", "train.jsonl")
TOOLS = os.path.join(LOCAL, "data_s2_tools_v3", "train.jsonl")
OUT_DIR = os.path.join(LOCAL, "data_s2_full", "B_mix_80_20")
SEED = 7
TOOLS_SHARE = 0.20

os.makedirs(OUT_DIR, exist_ok=True)

speech = [json.loads(l) for l in open(SPEECH, encoding="utf-8")]
tools = [json.loads(l) for l in open(TOOLS, encoding="utf-8")]
assert all("tool" not in r for r in speech), "speech rows must not have 'tool'"
assert all("tool" in r for r in tools), "tools rows must have 'tool'"

n_speech = len(speech)
n_tools_target = round(n_speech * TOOLS_SHARE / (1 - TOOLS_SHARE))
print(f"speech={n_speech} tools_pool={len(tools)} tools_target={n_tools_target}")

by_op = {}
for r in tools:
    by_op.setdefault(r["tool"], []).append(r)
ops = sorted(by_op)
pool_total = sum(len(v) for v in by_op.values())

exact = {op: n_tools_target * len(by_op[op]) / pool_total for op in ops}
alloc = {op: int(exact[op]) for op in ops}
rem = n_tools_target - sum(alloc.values())
for op in sorted(ops, key=lambda o: exact[o] - alloc[o], reverse=True)[:rem]:
    alloc[op] += 1
print("alloc:", alloc, "sum:", sum(alloc.values()))

rng = random.Random(SEED)
sampled = []
for op in ops:
    idx = rng.sample(range(len(by_op[op])), alloc[op])
    sampled.extend(by_op[op][i] for i in idx)

combined = speech + sampled
rng.shuffle(combined)

out = os.path.join(OUT_DIR, "train.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for r in combined:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

cn = Counter(("tools" if "tool" in r else "speech") for r in combined)
n_tools = cn["tools"]
speech_h = sum(r.get("duration", 0.0) for r in combined if "tool" not in r) / 3600
tools_h = sum(r.get("duration", 0.0) for r in combined if "tool" in r) / 3600
report = {
    "seed": SEED,
    "speech_rows": cn["speech"],
    "tools_rows": n_tools,
    "tools_share_rows": n_tools / len(combined),
    "total_rows": len(combined),
    "alloc_by_op": alloc,
    "speech_hours_tgt": round(speech_h, 2),
    "tools_hours_tgt": round(tools_h, 2),
    "speech_file": SPEECH,
    "tools_file": TOOLS,
}
json.dump(report, open(os.path.join(OUT_DIR, "mix_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
print("B_MIX_DONE")
