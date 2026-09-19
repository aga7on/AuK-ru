"""s3 train-микс 70/30 (GATES §4): речь v2_after_identity + tools v3 (rebalanced).

Tools: volume_up/down по 3200 (весь доступный volume_up=2207), pitch_up/down по 3200,
speed_up/down по 1600, noise_add ИСКЛЮЧЁН (отрицательный snr_gain у всех вариантов).
Доля tools после отбора стремится к ~30% без искусственного обрезания речи.
Выход: data_s2_full/v3_s3_mix/train.jsonl + mix_report.json.
"""
import json
import os
import random
from collections import Counter

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_train")
SPEECH = os.path.join(LT, "data_s2_full", "v2_after_identity", "train.jsonl")
TOOLS = os.path.join(LT, "data_s2_tools_v3", "train.jsonl")
OUT = os.path.join(LT, "data_s2_full", "v3_s3_mix")

TARGETS = {"volume_up": 3200, "volume_down": 3200, "pitch_up": 3200, "pitch_down": 3200,
           "speed_up": 1600, "speed_down": 1600}
SEED = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    speech = [json.loads(l) for l in open(SPEECH, encoding="utf-8") if l.strip()]
    tools = [json.loads(l) for l in open(TOOLS, encoding="utf-8") if l.strip()]

    by_op = {}
    for t in tools:
        by_op.setdefault(t["meta"]["op"], []).append(t)
    for op in by_op:
        rng.shuffle(by_op[op])
    picked = []
    for op, n in TARGETS.items():
        pool = by_op.get(op, [])
        picked += pool[:n]
    rng.shuffle(picked)

    # val (speaker-disjoint) — уже готов отдельным файлом
    n_speech, n_tools = len(speech), len(picked)
    rows = speech + picked
    rng.shuffle(rows)

    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    speech_h = sum(r.get("duration", 0) for r in speech) / 3600
    tools_h = sum(r.get("duration", 0) for r in picked) / 3600
    ops = Counter(r["meta"]["op"] for r in picked)
    # invariants: no noise_add, no duplicate rows (fast check by json hash)
    assert "noise_add" not in ops
    seen = set()
    dups = 0
    for r in rows:
        k = json.dumps(r, sort_keys=True)
        if k in seen:
            dups += 1
        seen.add(k)
    report = {"speech_rows": n_speech, "tools_rows": n_tools, "total": len(rows),
              "tools_share": round(n_tools / len(rows), 4),
              "speech_hours": round(speech_h, 2), "tools_hours": round(tools_h, 2),
              "ops": dict(ops), "duplicates": dups, "seed": SEED}
    json.dump(report, open(os.path.join(OUT, "mix_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("INVARIANTS_OK" if dups == 0 and "noise_add" not in ops else "INVARIANTS_FAIL")


if __name__ == "__main__":
    main()
