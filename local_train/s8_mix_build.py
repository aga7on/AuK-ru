# -*- coding: utf-8 -*-
"""S8-микс (v8_s8_mix): neutral recovery поверх v7_s7_mix.

Дизайн (ROADMAP S8):
  - НЕ механический clone-boost (S6 показал бесполезность oversampling как такового);
  - вместо этого: паттерн-майнинг hard cases ВНУТРИ v7 (тексты с акцентами, уже в train-домене):
      * длинные слова (≥11 букв) — «ломает длинные слова»;
      * консонантные кластеры (ств/нств/рств/вств/нкт/ртр/…) — «путается на согласных»;
      * длинные фразы (≥8 слов) — «проглатывает части, меняет окончания»;
    выбранные строки дублируются ×3 (hard replay);
  - clone-строки дублируются ×1 дополнительно (clone replay, умеренно);
  - emo-строки и tool-строки v7 сохраняются 1:1 (emotion replay + upstream replay);
  - ВАЖНО: сами 41 текст из hard_cases_ru_v1.jsonl — held-out benchmark, в train НЕ попадают
    (проверено check_contamination.py: пересекаются только 3 исторических текста).

Выход: local_train/data_s2_full/v8_s8_mix/train.jsonl
Вал:   тот же v3_s3_mix/val.jsonl (619).
"""
import json
import os
import random
import re
from collections import Counter

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v8_s8_mix")
SEED = 8
CLUSTERS = re.compile(r"(ств|нств|рств|вств|нкт|ртр|здн|вств|мпл|нтр|ктр|вск|здк|жч|щн|чн[ыо])")


def quoted(txt):
    m = re.search(r"'([^']*)'", txt or "")
    return m.group(1) if m else None


def classify(r):
    u = [c for m in r["messages"] if m["role"] == "user" for c in m["content"] if isinstance(c, dict)]
    txt = next((c["text"] for c in u if c.get("type") == "text"), "")
    has_audio = any(c.get("type") == "audio" for c in u)
    if " tone:" in txt:
        return "emo", txt
    if "Reproduce the reference voice" in txt:
        return "clone", txt
    if has_audio:
        return "tool", txt
    return "tts", txt


def is_hard(t):
    q = quoted(t)
    if not q:
        return False
    clean = q.replace("+", "")
    words = re.findall(r"[а-яёa-z]+", clean.lower())
    if not words:
        return False
    if any(len(w) >= 11 for w in words):
        return True
    if any(CLUSTERS.search(w) for w in words):
        return True
    if len(words) >= 8:
        return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print("v7 rows:", len(rows))

    out = []
    kinds = Counter()
    hard = Counter()
    hours = 0.0
    for r in rows:
        kind, txt = classify(r)
        kinds[kind] += 1
        out.append(r)
        hours += float(r.get("duration") or 0)
        if kind in ("tts", "clone"):
            if is_hard(txt):
                hard[kind] += 1
                out += [r, r]  # ×3 итого
                hours += 2 * float(r.get("duration") or 0)
        if kind == "clone":
            out.append(r)  # clone replay ×2 итого
            hours += float(r.get("duration") or 0)

    rng = random.Random(SEED)
    rng.shuffle(out)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    emo_share = kinds["emo"] / len(out)
    print("out rows:", len(out))
    print("kinds v7:", dict(kinds))
    print("hard picked:", dict(hard))
    print("hours: %.2f" % (hours / 3600))
    print("emo_share of total rows: %.3f" % emo_share)
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
