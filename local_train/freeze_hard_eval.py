# -*- coding: utf-8 -*-
"""Заморозка hard_eval_v1: 241 текст становятся ИЗМЕРИТЕЛЬНЫМ ПРИБОРОМ навсегда.

Правила (по решению пользователя 20.09):
  1. hard_eval_v1 НЕ используется для обучения никогда — только для оценки;
  2. позиции, уже присутствующие в train-миксах (обнаружено: 1 точное + 9 4-граммных
     совпадений, унаследованы из v7-базы), помечаются `contaminated: true` и
     ИСКЛЮЧАЮТСЯ из гейта (иначе гейт измеряет запоминание, а не произношение);
  3. файл помечается FROZEN — изменения запрещены; новые тексты идут в hard_eval_v2.

Выход:
  local_train/hard_eval_v1/hard_eval_v1.jsonl       (все 241, с флагами)
  local_train/hard_eval_v1/hard_eval_v1_clean.jsonl (только не-заражённые — для гейта)
  local_train/hard_eval_v1/FROZEN.md                (манифест заморозки + sha256)
"""
import hashlib
import json
import os
import re
from collections import defaultdict

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru.jsonl")
OUT_DIR = os.path.join(AUK, "local_train", "hard_eval_v1")
TRAIN_MIXES = ["v5_s5_mix", "v7_s7_mix", "v8_s8_mix", "v9_s8b_mix", "v12_emo_mix", "v16_diction_mix"]


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "").replace("ё", "е")).strip()


def fourgrams(t):
    w = t.split()
    return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))} if len(w) >= 4 else set()


def collect_train():
    texts = set()
    grams = set()
    for name in TRAIN_MIXES:
        p = os.path.join(AUK, "local_train", "data_s2_full", name, "train.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            for m in r.get("messages", []):
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        mm = re.search(r"'([^']*)'", c["text"])
                        if mm:
                            t = norm(mm.group(1))
                            texts.add(t)
                            grams |= fourgrams(t)
    return texts, grams


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    train_texts, train_grams = collect_train()

    out, clean = [], []
    n_exact = n_gram = 0
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        t = norm(r["text"])
        g4 = fourgrams(t)
        exact = t in train_texts
        gram = bool(g4 & train_grams)
        contaminated = bool(exact or gram)
        n_exact += exact
        n_gram += gram and not exact
        rec = {"id": r["id"], "category": r["category"], "text": r["text"],
               "contaminated": contaminated,
               "contamination_reason": ("exact_in_train" if exact else ("4gram_in_train" if gram else None))}
        out.append(rec)
        by_cat[r["category"]][0] += 1
        if not contaminated:
            clean.append(rec)
            by_cat[r["category"]][1] += 1

    p_all = os.path.join(OUT_DIR, "hard_eval_v1.jsonl")
    p_clean = os.path.join(OUT_DIR, "hard_eval_v1_clean.jsonl")
    with open(p_all, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(p_clean, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha_all, sha_clean = sha256_file(p_all), sha256_file(p_clean)
    L = ["# hard_eval_v1 — ЗАМОРОЖЕННЫЙ измерительный набор (FROZEN)", "",
         "**Статус: FROZEN. Изменения содержимого запрещены.** Любые правки ломают",
         "сравнимость результатов между этапами. Новые тексты — только в `hard_eval_v2`.", "",
         f"- Всего позиций: **{len(out)}**",
         f"- Заражено (уже в train-миксах): **{sum(1 for r in out if r['contaminated'])}** "
         f"(exact {n_exact}, 4-gram {n_gram}) — ИСКЛЮЧЕНЫ из гейта",
         f"- Чистых для гейта: **{len(clean)}** (`hard_eval_v1_clean.jsonl`)", "",
         "## sha256 (контроль неизменности)", "",
         f"- `hard_eval_v1.jsonl`: `{sha_all}`",
         f"- `hard_eval_v1_clean.jsonl`: `{sha_clean}`", "",
         "## Состав по категориям (всего / чистых)", "",
         "| категория | всего | чистых |", "|---|---|---|"]
    for c, (a, b) in sorted(by_cat.items()):
        L.append(f"| {c} | {a} | {b} |")
    L += ["", "## Правила использования", "",
          "1. Обучать на этих текстах ЗАПРЕЩЕНО (они — измерительный прибор).",
          "2. Гейт S16 и далее считается по `hard_eval_v1_clean.jsonl` (231 позиция).",
          "3. Проверка перед каждым новым обучением: `check_hard_eval_freeze.py` (exact + 4-gram).",
          "4. Источники заражений — v7-база (строки clone100 TEXTS и разговорные фразы);",
          "   S16-добавка (clean_clips) проверена: 0 exact / 0 4-gram — чистая.", "",
          "## Метрики гейта (по решению пользователя, 20.09)", "",
          "- `human_mumble_rate` — доля файлов с человеческим флагом «жуёт слова»;",
          "- `both_bad_rate` — доля пар A/B с вердиктом «оба плохи»;",
          "- ошибки ы/и; ошибки hard/soft (палатализация); пропуски фонем/слогов;",
          "- неправильные ударения; repetitions/stumbles; hard-text text_fidelity.",
          "- Gemini-судья — ТОЛЬКО triage/shortlist, НЕ финальный вердикт",
          "  (согласованность с человеком 21%, S13_CALIBRATION.md)."]
    open(os.path.join(OUT_DIR, "FROZEN.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    print(f"frozen: {len(out)} total, {len(clean)} clean, contaminated {sum(1 for r in out if r['contaminated'])}")
    print("sha_all:", sha_all[:16], "sha_clean:", sha_clean[:16])
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
