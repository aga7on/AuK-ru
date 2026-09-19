# -*- coding: utf-8 -*-
"""S10 RFT-микс: preference-пары → train.jsonl (chosen как target) + replay v9.

Формат строки — как в v7/v9 (user: instruction+ref audio, assistant: chosen wav).
Инструкция реконструируется по типу пары:
  - clone-пары (mined/mined2): «Reproduce the reference voice and say in Russian: '<text>'»;
  - эмо-пары (emotion_ru_*): «... with a <emotion> tone: '<text>'».

Состав: все chosen из pairs_*.jsonl (×REPEAT, по умолчанию 4) + replay из v9_s8b_mix
(доля REPLAY_SHARE, по умолчанию 0.5 — защита от забывания; короткий прогон 500 шагов).

usage: python s10_rft_mix.py [--repeat 4] [--replay-share 0.5] [--out ...]
Выход: local_train/data_s2_full/v10_s10_rft/train.jsonl
"""
import argparse
import json
import os
import random
from collections import Counter

AUK = r"G:\AI\AuK"
PREF = os.path.join(AUK, "local_train", "preference")
V9 = os.path.join(AUK, "local_train", "data_s2_full", "v9_s8b_mix", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v10_s10_rft")
SEED = 10


def pair_rows(repeat):
    rows = []
    for name in ("pairs_v1.jsonl", "pairs_mined_v1.jsonl", "pairs_mined_v2.jsonl"):
        p = os.path.join(PREF, name)
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            chosen = r["chosen"]
            if not os.path.exists(chosen):
                continue
            import soundfile as sf
            try:
                dur = float(sf.info(chosen).duration)
            except Exception:
                continue
            text = r.get("text") or ""
            emo = r.get("emotion")
            if not emo and str(r.get("source", "")).startswith("emotion"):
                parts = str(r.get("prompt_id", "")).split("_")
                emo = parts[1] if len(parts) > 1 and parts[1] in (
                    "happy", "sad", "angry", "fearful", "excited") else None
            if emo:
                instr = f"Reproduce the reference voice and say in Russian with a {emo} tone: '{text}'"
            else:
                instr = f"Reproduce the reference voice and say in Russian: '{text}'"
            row = {"duration": round(dur, 3),
                   "messages": [
                       {"role": "user", "content": [
                           {"type": "text", "text": instr},
                           {"type": "audio", "audio": r["ref"]}]},
                       {"role": "assistant", "content": [
                           {"type": "audio", "audio_url": chosen}]}],
                   "split": "train",
                   "meta": {"src": name, "pair_id": r["pair_id"], "metric": r.get("metric"),
                            "chosen_score": r.get("chosen_score"), "rejected_score": r.get("rejected_score")}}
            rows += [row] * repeat
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--replay-share", type=float, default=0.5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pref = pair_rows(args.repeat)
    print("preference rows:", len(pref))
    n_replay = int(len(pref) * args.replay_share / (1 - args.replay_share)) if args.replay_share < 1 else len(pref)

    rng = random.Random(SEED)
    v9 = [json.loads(l) for l in open(V9, encoding="utf-8") if l.strip()]
    replay = rng.sample(v9, min(n_replay, len(v9)))
    out = pref + replay
    rng.shuffle(out)

    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration") or 0) for r in out) / 3600
    srcs = Counter(r.get("meta", {}).get("src", "replay_v9") for r in out)
    print("total rows:", len(out), "| pref:", len(pref), "| replay:", len(replay))
    print("hours: %.2f" % hours)
    print("sources:", dict(srcs))
    print("wrote", os.path.join(args.out, "train.jsonl"))


if __name__ == "__main__":
    main()
