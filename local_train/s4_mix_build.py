"""s4 train-микс: речь v2_after_identity + tools_v3 (половина) + tools_v4 (все 5600).

Цель s4: починить перелёт величины (volume_up +12 вместо +6, pitch +3.9 вместо +2),
сохранив распределение ops из s3. v3-цели оставляют ровно одну величину на op —
v4 добавляет разброс +3/+6/+9 дБ, ±1/2/3 st, x1.1/1.2/0.9/0.8.
Tools: v3 volume_up 2207 (весь), v3 volume_down/pitch 1600, v3 speed 800, v4 5600.
Выход: data_s2_full/v4_s4_mix/train.jsonl (val = тот же, что у s3).
"""
import json
import os
import random
from collections import Counter

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_train")
SPEECH = os.path.join(LT, "data_s2_full", "v2_after_identity", "train.jsonl")
V3 = os.path.join(LT, "data_s2_tools_v3", "train.jsonl")
V4 = os.path.join(LT, "data_s2_tools_v4", "train.jsonl")
OUT = os.path.join(LT, "data_s2_full", "v4_s4_mix")

V3_TARGETS = {"volume_up": 2207, "volume_down": 1600, "pitch_up": 1600, "pitch_down": 1600,
              "speed_up": 800, "speed_down": 800}
SEED = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    speech = [json.loads(l) for l in open(SPEECH, encoding="utf-8") if l.strip()]
    v3 = [json.loads(l) for l in open(V3, encoding="utf-8") if l.strip()]
    v4 = [json.loads(l) for l in open(V4, encoding="utf-8") if l.strip()]

    by_op = {}
    for t in v3:
        by_op.setdefault(t["meta"]["op"], []).append(t)
    for op in by_op:
        rng.shuffle(by_op[op])
    v3_picked = []
    for op, n in V3_TARGETS.items():
        v3_picked += by_op.get(op, [])[:n]

    # в v4 скорость уже двух величин — берём всё
    tools = v3_picked + v4
    rows = speech + tools
    rng.shuffle(rows)

    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    speech_h = sum(r.get("duration", 0) for r in speech) / 3600
    tools_h = sum(r.get("duration", 0) for r in tools) / 3600
    ops = Counter(r["meta"]["op"] for r in tools)
    assert "noise_add" not in ops
    seen, dups = set(), 0
    for r in rows:
        k = json.dumps(r, sort_keys=True)
        if k in seen:
            dups += 1
        seen.add(k)
    report = {"speech_rows": len(speech), "v3_rows": len(v3_picked), "v4_rows": len(v4),
              "total": len(rows), "tools_share": round(len(tools) / len(rows), 4),
              "speech_hours": round(speech_h, 2), "tools_hours": round(tools_h, 2),
              "ops": dict(ops), "duplicates": dups, "seed": SEED}
    json.dump(report, open(os.path.join(OUT, "mix_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("INVARIANTS_OK" if dups == 0 and "noise_add" not in ops else "INVARIANTS_FAIL")


if __name__ == "__main__":
    main()
