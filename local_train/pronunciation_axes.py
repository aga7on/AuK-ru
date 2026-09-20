# -*- coding: utf-8 -*-
"""Сквозной анализ ОСЕЙ ПРОИЗНОШЕНИЯ во всех накопленных результатах судьи v3.

Схема TTS/phonetics содержит: accent (10 = нейтральный русский), palatalization
(мягкость перед и/е/ё/ю/я), stress (ударения), endings (окончания), text_fidelity,
phoneme_substitutions (формат «ы/и», «ч/ц», «ш/щ» — ровно то, что услышал пользователь).

Скрипт собирает эти оси по всем *_judge_results.jsonl в reports/deepseek_supervised,
группирует по чекпойнту (из имени файла) и группе (tts/phonetics), ищет упоминания
«китайск» в issues и топы фонемных подмен.

usage: python pronunciation_axes.py [--min_n 5]
Выход: stdout + reports/deepseek_supervised/PRONUNCIATION_AXES.md
"""
import json
import os
import re
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
OUT = os.path.join(D, "PRONUNCIATION_AXES.md")
AXES = ("accent", "palatalization", "stress", "endings", "text_fidelity", "naturalness", "prosody")

# имя файла → человекочитаемый чекпойнт
TAGS = [
    ("s5_4500", "s5@4500"), ("s7_5750", "s7@5750 (v1.0)"), ("s8_7750", "s8@7750"),
    ("s8b_8500", "s8b@8500"), ("s10rft", "s10rft@6500"), ("s12_7250", "s12@7250"),
    ("s13_phonetics", "A/B blind (s7 vs s8b)"), ("emotion_s7_5750", "s7@5750 emo"),
    ("emotion_s8b_8500", "s8b@8500 emo"), ("emotion_s12_7250", "s12@7250 emo"),
    ("clone100", "clone100"),
]


def jl(p):
    try:
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    except Exception:
        return []


def main():
    files = [f for f in os.listdir(D) if f.endswith("_judge_results.jsonl") or f.endswith("_results.jsonl")]
    agg = defaultdict(lambda: defaultdict(list))   # tag -> group -> [judge dicts]
    subs = Counter()
    mangled = Counter()
    chinese = Counter()
    issues_hits = []

    for fn in sorted(files):
        tag = None
        for key, label in TAGS:
            if key in fn:
                tag = label
                break
        if tag is None:
            tag = fn.replace("_judge_results.jsonl", "").replace("_results.jsonl", "")
        for r in jl(os.path.join(D, fn)):
            if r.get("status") != "ok":
                continue
            j = r.get("judge") or {}
            grp = r.get("group", "?")
            if not any(isinstance(j.get(a), int) for a in AXES):
                continue  # не TTS-схема (clone/tool/OP) — осей произношения нет
            agg[tag][grp].append(j)
            for s in j.get("phoneme_substitutions") or []:
                subs[str(s)] += 1
            for w in j.get("words_mangled") or []:
                mangled[str(w).lower()] += 1
            iss = str(j.get("issues") or "")
            if "китайск" in iss.lower():
                chinese[tag] += 1
                if len(issues_hits) < 12:
                    issues_hits.append((tag, r.get("task_id"), iss[:160]))

    def m(js, k):
        v = [j.get(k) for j in js if isinstance(j.get(k), int)]
        return round(sum(v) / len(v), 2) if v else None

    def low(js, k, thr=6):
        v = [j.get(k) for j in js if isinstance(j.get(k), int)]
        return sum(1 for x in v if x < thr) if v else None

    L = ["# PRONUNCIATION AXES — сквозной анализ всех прогонов судьи v3", "",
         "Оси схемы TTS/phonetics: accent (10 = нейтральный русский), palatalization (мягкость",
         "перед и/е/ё/ю/я), stress (ударения), endings (окончания). Значения 0–10, выше = лучше.",
         "Порог «проблема» = оценка < 6.", "",
         "## Средние по осям (группа tts; n в скобках)", "",
         "| прогон | n | accent | palatalization | stress | endings | text_fidelity |",
         "|---|---|---|---|---|---|---|"]
    rows = []
    for tag, groups in agg.items():
        for grp, js in groups.items():
            if len(js) < 3:
                continue
            rows.append((tag, grp, js))
    for tag, grp, js in sorted(rows, key=lambda x: (x[1], x[0])):
        L.append(f"| {tag} [{grp}] | {len(js)} | {m(js,'accent')} | {m(js,'palatalization')} | "
                 f"{m(js,'stress')} | {m(js,'endings')} | {m(js,'text_fidelity')} |")

    L += ["", "## Доля файлов с оценкой < 6 (проблемные)", "",
          "| прогон | n | accent<6 | palat<6 | stress<6 | endings<6 |", "|---|---|---|---|---|---|"]
    for tag, grp, js in sorted(rows, key=lambda x: (x[1], x[0])):
        L.append(f"| {tag} [{grp}] | {len(js)} | {low(js,'accent')} | {low(js,'palatalization')} | "
                 f"{low(js,'stress')} | {low(js,'endings')} |")

    L += ["", "## Упоминания «китайского» акцента в issues судьи", ""]
    if chinese:
        L.append("| прогон | n |")
        L.append("|---|---|")
        for t, n in chinese.most_common():
            L.append(f"| {t} | {n} |")
    else:
        L.append("Судья НИ РАЗУ не упомянул «китайский акцент» ни в одном прогоне (0 упоминаний).")

    L += ["", "## Топ фонемных подмен (судья)", ""]
    if subs:
        L.append("| подмена | n |")
        L.append("|---|---|")
        for s, n in subs.most_common(20):
            L.append(f"| {s} | {n} |")
    else:
        L.append("Судья не вернул ни одной фонемной подмены во всех прогонах (пустые списки).")

    L += ["", "## Топ прожёванных слов (words_mangled)", ""]
    if mangled:
        L.append("| слово | n |")
        L.append("|---|---|")
        for w, n in mangled.most_common(20):
            L.append(f"| {w} | {n} |")
    else:
        L.append("(пусто)")

    L += ["", "## Примеры issues с «китайск»", ""]
    if issues_hits:
        for t, tid, iss in issues_hits:
            L.append(f"- `{t}` / {tid}: {iss}")
    else:
        L.append("(нет)")

    L += ["", "## Интерпретация", "",
          "Заполняется по таблицам: если accent/palatalization/stress в среднем высокие (≥8) и",
          "подмен нет — судья НЕ подтверждает наблюдение пользователя; расхождение «человек слышит,",
          "судья не слышит» само по себе результат (предел автосудьи по тонкой фонетике).",
          "Если оси низкие — подтверждено, и это новый топ-приоритет."]

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("runs analyzed:", len(rows))
    for tag, grp, js in sorted(rows, key=lambda x: (x[1], x[0])):
        print(f"  {tag} [{grp}] n={len(js)} accent={m(js,'accent')} palat={m(js,'palatalization')} "
              f"stress={m(js,'stress')} endings={m(js,'endings')}")
    print("chinese mentions:", dict(chinese))
    print("top subs:", subs.most_common(8))
    print("top mangled:", mangled.most_common(8))


if __name__ == "__main__":
    main()
