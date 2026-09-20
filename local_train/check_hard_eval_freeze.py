# -*- coding: utf-8 -*-
"""КРИТИЧЕСКАЯ проверка: замороженный eval-набор hard_eval_v1 (241 текст) НЕ должен
встречаться ни в одном train-миксе. Проверяет и точные тексты, и подстроки/словоформы.

usage: python check_hard_eval_freeze.py
"""
import json
import os
import re
from collections import Counter

AUK = r"G:\AI\AuK"
FROZEN = os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru.jsonl")
MIXES = ["v5_s5_mix", "v7_s7_mix", "v8_s8_mix", "v9_s8b_mix", "v12_emo_mix", "v16_diction_mix"]


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "").replace("ё", "е")).strip()


def main():
    frozen = [json.loads(l) for l in open(FROZEN, encoding="utf-8") if l.strip()]
    texts = [norm(r["text"]) for r in frozen]
    print(f"FROZEN hard_eval_v1: {len(texts)} texts (unique {len(set(texts))})")

    # 1) точные совпадения нормализованного текста
    # 2) совпадения по 4-граммам слов (чтобы ловить «та же фраза, другое окончание»)
    def fourgrams(t):
        w = t.split()
        return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))} if len(w) >= 4 else set()

    frozen_set = set(texts)
    frozen_4g = set()
    for t in texts:
        frozen_4g |= fourgrams(t)

    for name in MIXES:
        p = os.path.join(AUK, "local_train", "data_s2_full", name, "train.jsonl")
        if not os.path.exists(p):
            print(f"{name}: MISSING")
            continue
        train_texts = set()
        train_4g = set()
        n = 0
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            n += 1
            for m in r.get("messages", []):
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        mm = re.search(r"'([^']*)'", c["text"])
                        if mm:
                            t = norm(mm.group(1))
                            train_texts.add(t)
                            train_4g |= fourgrams(t)
        exact = frozen_set & train_texts
        g4 = frozen_4g & train_4g
        print(f"{name}: rows={n} train_texts={len(train_texts)} | EXACT match={len(exact)} | 4-gram match={len(g4)}")
        for t in list(exact)[:5]:
            print("   EXACT LEAK:", t[:90])
        for g in list(g4)[:3]:
            print("   4gram LEAK:", " ".join(g)[:90])

    # дополнительно: v16 clean_clips (источник S16-добавки)
    cc = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "clean_clips.jsonl")
    if os.path.exists(cc):
        rows = [json.loads(l) for l in open(cc, encoding="utf-8") if l.strip()]
        ct = set(norm(r["text"]) for r in rows)
        c4 = set()
        for t in ct:
            c4 |= fourgrams(t)
        print(f"v16 clean_clips: {len(rows)} clips | EXACT match with frozen={len(frozen_set & ct)}"
              f" | 4-gram match={len(frozen_4g & c4)}")


if __name__ == "__main__":
    main()
