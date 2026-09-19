# -*- coding: utf-8 -*-
"""S8b-микс (v9_s8b_mix): точечная правка по итогам S8 (S8_RESULTS.md → Решение).

Изменения против v8_s8_mix:
  1. hard-case replay: ×3 → ×2 и ТОЛЬКО фонетические паттерны (длинные слова ≥11 букв,
     консонантные кластеры); критерий «длинная фраза ≥8 слов» убран (он тащил number/money
     тексты — они уже усвоены: TTS WER 0.069);
  2. clone-дублирование ×2 → ×1 (откат к v7: S6+S8 доказали бесполезность oversampling);
  3. +3000 строк neutral TTS replay из v5_s5_mix (плоский tts без emo/ref — восстановление
     first_ok до ≥0.812);
  4. emo/tool сохраняются 1:1 (эмоции 48.3% не трогать).

Старт обучения: s8@7750 (model_last), lr 3e-6, 750 шагов → цель 8500.
Выход: local_train/data_s2_full/v9_s8b_mix/train.jsonl
"""
import json
import os
import random
import re
from collections import Counter

AUK = r"G:\AI\AuK"
SRC_V7 = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
SRC_V5 = os.path.join(AUK, "local_train", "data_s2_full", "v5_s5_mix", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v9_s8b_mix")
SEED = 9
NEUTRAL_REPLAY = 3000
CLUSTERS = re.compile(r"(ств|нств|рств|вств|нкт|ртр|здн|мпл|нтр|ктр|вск|здк|жч|щн)")


def quoted(txt):
    m = re.search(r"'([^']*)'", txt or "")
    return m.group(1) if m else None


def classify(r):
    u = [c for m in r["messages"] if m["role"] == "user" for c in m["content"] if isinstance(c, dict)]
    txt = next((c["text"] for c in u if c.get("type") == "text"), "")
    has_audio = any(c.get("type") == "audio" for c in u)
    if " tone:" in txt:
        return "emo", txt, has_audio
    if "Reproduce the reference voice" in txt:
        return "clone", txt, has_audio
    if has_audio:
        return "tool", txt, has_audio
    return "tts", txt, has_audio


def is_phonetic_hard(t):
    """Только фонетические паттерны: длинные слова и консонантные кластеры."""
    q = quoted(t)
    if not q:
        return False
    words = re.findall(r"[а-яёa-z]+", q.replace("+", "").lower())
    if not words:
        return False
    return any(len(w) >= 11 for w in words) or any(CLUSTERS.search(w) for w in words)


def main():
    os.makedirs(OUT, exist_ok=True)
    v7 = [json.loads(l) for l in open(SRC_V7, encoding="utf-8") if l.strip()]
    v5 = [json.loads(l) for l in open(SRC_V5, encoding="utf-8") if l.strip()]
    print("v7 rows:", len(v7), "v5 rows:", len(v5))

    out = []
    kinds = Counter()
    hard_n = 0
    hours = 0.0
    for r in v7:
        kind, txt, _ = classify(r)
        kinds[kind] += 1
        out.append(r)
        hours += float(r.get("duration") or 0)
        # NO clone duplication (S8b fix #2)
        if kind == "tts" and is_phonetic_hard(txt):
            hard_n += 1
            out.append(r)  # ×2 total (S8b fix #1)
            hours += float(r.get("duration") or 0)

    # neutral TTS replay from v5 (плоские tts-строки, без emo/clone/tool)
    v5_tts = [r for r in v5 if classify(r)[0] == "tts"]
    rng = random.Random(SEED)
    rng.shuffle(v5_tts)
    replay = v5_tts[:NEUTRAL_REPLAY]
    print("v5 tts pool:", len(v5_tts), "replay picked:", len(replay))
    out += replay
    hours += sum(float(r.get("duration") or 0) for r in replay)
    kinds["tts"] += len(replay)

    rng.shuffle(out)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("out rows:", len(out))
    print("kinds:", dict(kinds))
    print("phonetic-hard duplicated:", hard_n)
    print("hours: %.2f" % (hours / 3600))
    print("emo share: %.3f" % (kinds["emo"] / len(out)))
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
