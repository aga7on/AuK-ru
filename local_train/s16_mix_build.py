# -*- coding: utf-8 -*-
"""S16-микс (v16_diction_mix): фонетический буткемп на ПРОВЕРЕННО чистой дикции.

Дизайн (учитывает провал S8: общий hard-replay сломал first_ok):
  - база: v7_s7_mix 1:1 (сохраняет ВСЁ, что умеет v1.0 — эмоции, tools, clone);
  - добавка: clean_clips.jsonl (GigaAM WER ≤ 0.10, без плохих подмен) как TTS-строки,
    дублированные ×2 (умеренно — НЕ ×3 как в S8);
  - приоритет клипам с трудными словами (сегодня/университета/холодильник/…) —
    «трудное слово, произнесённое чисто» = золотой пример;
  - стоп-условие в обучении: TTS first_ok не ниже 0.812 (гейт S16).

Формат строк — как в v7 TTS (инструкция «Say the following in Russian with clear,
natural pronunciation: '<текст с ударениями>'» + assistant audio_url).

usage: python s16_mix_build.py [--repeat 2] [--max_add 3000]
Выход: local_train/data_s2_full/v16_diction_mix/train.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, r"G:\AI\AuK\src")

AUK = r"G:\AI\AuK"
SRC_V7 = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
CLEAN = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "clean_clips.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction_mix")
SEED = 16

INSTR = "Say the following in Russian with clear, natural pronunciation: '{}'"

# ОПЦИОНАЛЬНЫЙ фильтр контента (--filter-profanity, по умолчанию ВЫКЛЮЧЕН).
# Решение пользователя (20.09): для фонетического буткемпа фильтр ВРЕДЕН —
#  (1) TTS произносит то, что ему дали; удаление слов из train не запрещает их озвучить,
#      а лишь ухудшает их произношение (прямой конфликт с целью S16);
#  (2) регэкст дал ложные срабатывания на легитимном контенте (новостное предложение
#      «…покончить с сексуальным насилием…» было отброшено как «непристойное»);
#  (3) корпус kyutai-ru разговорный — вырезание мата создаёт распределенческий перекос;
#  (4) если фильтрация контента нужна, её место — frontend/инструкция, а не train-данные.
BAD_RE = re.compile(
    r"(ху[йяюеи]|бл[яa][дt]|пизд|еба|ёба|заеб|наху|сук[аи]|муда|гандон|пид[оa]р|"
    r"дерьм|жоп[ауы]|член|секс|траха|шлюх|херн|поху|охуе|долбоеб|ебать)", re.I)


def is_clean_content(text):
    return BAD_RE.search(text or "") is None




def accentize(text):
    """Расстановка ударений через штатный frontend (RUAccent), как в train-строках v7."""
    try:
        from auk.infer.infer_gradio import _accentize_ru
        return _accentize_ru(text, use_lexicon=False)
    except Exception:
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max_add", type=int, default=3000)
    ap.add_argument("--filter-profanity", action="store_true",
                    help="НЕ рекомендуется: ухудшает произношение отфильтрованных слов "
                         "и даёт ложные срабатывания на легитимном контенте")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    v7 = [json.loads(l) for l in open(SRC_V7, encoding="utf-8") if l.strip()]
    clean = [json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip()]
    print("v7 rows:", len(v7), "| clean clips:", len(clean))

    # уже присутствующие аудио в v7 — не дублировать
    existing = set()
    for r in v7:
        for m in r["messages"]:
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "audio":
                    u = c.get("audio") or c.get("audio_url")
                    if u:
                        existing.add(os.path.normcase(u))
    fresh = [c for c in clean if os.path.normcase(c["path"]) not in existing]
    n_before = len(fresh)
    if args.filter_profanity:
        fresh = [c for c in fresh if is_clean_content(c.get("text"))]
        print("clean clips NOT already in v7:", n_before,
              "| after profanity filter:", len(fresh), f"(removed {n_before - len(fresh)})")
    else:
        print("clean clips NOT already in v7:", n_before, "(profanity filter OFF — по решению пользователя)")

    # приоритет: с трудными словами
    fresh.sort(key=lambda c: (-len(c.get("hard_words") or []), c["wer"]))
    picked = fresh[: args.max_add]
    print("picked:", len(picked), "| with hard words:",
          sum(1 for c in picked if c.get("hard_words")))

    rng = random.Random(SEED)
    added = []
    for c in picked:
        text = accentize(c["text"].strip())
        row = {"duration": float(c["dur"]),
               "messages": [
                   {"role": "user", "content": [{"type": "text", "text": INSTR.format(text)}]},
                   {"role": "assistant", "content": [{"type": "audio", "audio_url": c["path"]}]}],
               "split": "train",
               "meta": {"source": "s16_clean_diction", "wer": c["wer"],
                        "hard_words": c.get("hard_words") or []}}
        added += [row] * args.repeat

    out = v7 + added
    rng.shuffle(out)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration") or 0) for r in out) / 3600
    kinds = Counter()
    for r in out:
        t = None
        for m in r["messages"]:
            if m["role"] == "user":
                for cc in m["content"]:
                    if isinstance(cc, dict) and cc.get("type") == "text":
                        t = cc["text"]
        if t and " tone:" in t:
            kinds["emo"] += 1
        elif (r.get("meta") or {}).get("source") == "s16_clean_diction":
            kinds["s16_clean"] += 1
        else:
            kinds["other"] += 1
    print("out rows:", len(out), "| hours: %.1f" % hours)
    print("kinds:", dict(kinds))
    print("added share: %.3f" % (len(added) / len(out)))
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
