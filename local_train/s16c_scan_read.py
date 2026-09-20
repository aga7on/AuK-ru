# -*- coding: utf-8 -*-
"""S16c step 1: скан ЧИТАНОЙ речи (Common Voice) на покрытие шести корзин.

Структурная находка (21.09): в корпусе два пула —
  ru_wav     (87 039 файлов): 32 259 = common_voice_ru_* (ЧИТАНАЯ речь) + спонтанные;
  ru_wav_mfa (942 026 файлов): 0 common_voice → ПОЛНОСТЬЮ спонтанная речь.
Предыдущий deficit-скан шёл по corpus_index (ru_wav_mfa) → 100% разговорки → S16b
обучался имитировать разговорный стиль (13% филлеров), что и сломало hard-гейт.

Этот скрипт: берёт ТОЛЬКО common_voice_* из ru_all.jsonl, фильтрует разговорные маркеры,
классифицирует по корзинам (переиспользуя phonetic_baskets), защищает frozen hard_eval_v1.
ASR-верификация — отдельным шагом (s16_verify_deficit-подобным).

usage: python s16c_scan_read.py [--out read_candidates.jsonl]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
RU_ALL = r"G:\AI\kyutai-ru\data\ru_all.jsonl"
RU_WAV = r"G:\AI\kyutai-ru\data\ru_wav"
FROZEN = os.path.join(AUK, "local_train", "hard_eval_v1", "hard_eval_v1.jsonl")
V7 = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
OUT_DIR = os.path.join(AUK, "local_train", "data_s2_full", "v16c_read")

from phonetic_baskets import baskets, norm, fourgrams  # noqa: E402

# разговорные маркеры: читаная речь их содержать не должна
FILLER = re.compile(r"\b(ну|нуу|как бы|типа|короче|вот|это самое|блин|значит|вообще|"
                    r"допустим|слушай|понимаешь|знаешь|короч|воот|ээ|мм)\b")
SELF_REPAIR = re.compile(r"(\w{3,})\s+\1")  # «лучше я луч…»


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "read_candidates.jsonl"))
    ap.add_argument("--min_words", type=int, default=6)
    ap.add_argument("--max_dur", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    frozen_4g = set()
    for l in open(FROZEN, encoding="utf-8"):
        frozen_4g |= fourgrams(norm(json.loads(l)["text"]))

    v7_audio = set()
    for l in open(V7, encoding="utf-8"):
        r = json.loads(l)
        for m in r["messages"]:
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "audio":
                    u = c.get("audio") or c.get("audio_url")
                    if u:
                        v7_audio.add(os.path.normcase(u))

    stats = Counter()
    by_basket = defaultdict(list)
    out = []
    for l in open(RU_ALL, encoding="utf-8"):
        r = json.loads(l)
        p = r.get("path")
        t = (r.get("transcript") or "").strip()
        d = float(r.get("duration") or 0)
        if not p or not t:
            continue
        if not os.path.basename(p).startswith("common_voice"):
            stats["not_read_speech"] += 1
            continue
        stats["read_total"] += 1
        if not (2.0 <= d <= args.max_dur):
            stats["bad_duration"] += 1
            continue
        if len(t.split()) < args.min_words:
            stats["too_short"] += 1
            continue
        if FILLER.search(t.lower()) or SELF_REPAIR.search(t.lower()):
            stats["conversational_markers"] += 1
            continue
        # файл может лежать в ru_wav (ru_all указывает ru_wav)
        cand = p if os.path.exists(p) else os.path.join(RU_WAV, os.path.basename(p))
        if not os.path.exists(cand) or os.path.getsize(cand) < 1000:
            stats["missing_file"] += 1
            continue
        if os.path.normcase(cand) in v7_audio:
            stats["already_in_v7"] += 1
            continue
        if fourgrams(norm(t)) & frozen_4g:
            stats["frozen_overlap"] += 1
            continue
        bs = baskets(t)
        bs.discard("general")
        if not bs:
            stats["no_basket"] += 1
            continue
        rec = {"path": cand, "text": t, "dur": round(d, 2), "baskets": sorted(bs)}
        out.append(rec)
        for b in bs:
            by_basket[b].append(rec)
        stats["accepted"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("stats:", dict(stats))
    print("accepted:", len(out))
    hours = sum(r["dur"] for r in out) / 3600
    print("hours: %.1f" % hours)
    print("basket coverage:")
    for b in ("yi", "softness", "clusters", "devoicing", "stress", "long_num", "topwords"):
        n = len(by_basket.get(b, []))
        h = sum(x["dur"] for x in by_basket.get(b, [])) / 3600
        print(f"  {b:12s}: {n:6d} clips  {h:5.2f} h")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
