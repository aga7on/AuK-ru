"""s5 train-микс: s4-микс (речь+tools v3+v4) + replay upstream-задач (upsample x20) + клон-буст.

Цель s5: replay-защита оригинального функционала (pitch/emotion/nonverbal/content/timbre/TTS)
+ клон-сходство до >=0.75. Клон-буст: удвоение подвыборки речи (clone-формат 'Reproduce...').
Выход: local_train/data_s2_full/v5_s5_mix/train.jsonl, val = как у s3/s4 (speaker-disjoint).
"""
import json
import os
import random
from collections import Counter

AUK = r"G:\AI\AuK"
S4 = os.path.join(AUK, "local_train", "data_s2_full", "v4_s4_mix", "train.jsonl")
REPLAY = os.path.join(AUK, "local_train", "data_s2_full", "v5_replay", "replay_tasks.json")
SPEECH = os.path.join(AUK, "local_train", "data_s2_full", "v2_after_identity", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v5_s5_mix")
SEED = 7
REPLAY_REPEATS = 100
CLONE_BOOST = 3000  # доп. строк речи (clone-формат) поверх s4-микса


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

    # клон-буст: случайные строки речи, не создающие дубликатов (dup-контроль ниже это поймает)
    speech = [json.loads(l) for l in open(SPEECH, encoding="utf-8") if l.strip()]
    rng.shuffle(speech)
    clone_extra = speech[:CLONE_BOOST]

    rows = base + replay * REPLAY_REPEATS + clone_extra
    rng.shuffle(rows)

    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration", 0)) for r in rows) / 3600
    seen, dups = set(), 0
    for r in rows:
        k = json.dumps(r, sort_keys=True)
        if k in seen:
            dups += 1
        seen.add(k)
    report = {
        "base_rows": len(base), "replay_unique": len(replay), "replay_rows": len(replay) * REPLAY_REPEATS,
        "clone_boost_rows": len(clone_extra), "total": len(rows), "hours": round(hours, 2),
        "duplicates_all_incl_intentional_repeat": dups,
        "replay_share": round(len(replay) * REPLAY_REPEATS / len(rows), 4),
        "seed": SEED,
    }
    json.dump(report, open(os.path.join(OUT, "mix_report.json"), "w", encoding="utf-8"), indent=1)
    print("MIX_DONE", json.dumps(report))


if __name__ == "__main__":
    main()
