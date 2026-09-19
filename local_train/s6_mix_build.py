"""s6 train-микс: как s5, но клон-буст удвоен до 6000 строк (цель: clone sim >=0.75).
Выход: local_train/data_s2_full/v6_s6_mix/train.jsonl. Replay x100 сохранён.
"""
import json
import os
import random

AUK = r"G:\AI\AuK"
S4 = os.path.join(AUK, "local_train", "data_s2_full", "v4_s4_mix", "train.jsonl")
REPLAY = os.path.join(AUK, "local_train", "data_s2_full", "v5_replay", "replay_tasks.json")
SPEECH = os.path.join(AUK, "local_train", "data_s2_full", "v2_after_identity", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v6_s6_mix")
SEED = 7
REPLAY_REPEATS = 100
CLONE_BOOST = 6000


def replay_row(t):
    return {
        "duration": float(t["duration"]),
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": t["instruction"]},
                {"type": "audio", "audio": t["ref"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "audio", "audio_url": t["target"]},
            ]},
        ],
        "split": "train",
        "meta": {"replay": True, "fixture": t["fixture"]},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    base = [json.loads(l) for l in open(S4, encoding="utf-8") if l.strip()]
    replay = [replay_row(t) for t in json.load(open(REPLAY, encoding="utf-8"))]
    speech = [json.loads(l) for l in open(SPEECH, encoding="utf-8") if l.strip()]
    rng.shuffle(speech)
    clone_extra = speech[:CLONE_BOOST]

    rows = base + replay * REPLAY_REPEATS + clone_extra
    rng.shuffle(rows)

    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration", 0)) for r in rows) / 3600
    bad = sum(1 for r in rows if not isinstance(r.get("duration"), float))
    report = {
        "base_rows": len(base), "replay_unique": len(replay),
        "replay_rows": len(replay) * REPLAY_REPEATS,
        "clone_boost_rows": len(clone_extra), "total": len(rows),
        "hours": round(hours, 2), "non_float_duration": bad,
        "replay_share": round(len(replay) * REPLAY_REPEATS / len(rows), 4),
        "seed": SEED,
    }
    json.dump(report, open(os.path.join(OUT, "mix_report.json"), "w", encoding="utf-8"), indent=1)
    print("MIX_DONE", json.dumps(report))


if __name__ == "__main__":
    main()
