# -*- coding: utf-8 -*-
"""S13b: анализ произносительного прогона судьи (80 файлов A/B) → S13_PHONETICS.md.

Проверяет наблюдения пользователя: «клучи» вместо «ключи» (палатализация), «ми» вместо «мы»
(ы→и), потеря ударений, «китайский» акцент местами. Схема phonetics даёт оси:
text_fidelity, endings, naturalness, prosody, accent, palatalization, stress +
фонемные подмены (формат «ж/з, ч/ц, ы/и, ш/щ»).

Разбивка: по вариантам (s7/s8), по категориям пар (tts_hard/clone/emotion),
пары «жуёт»-флагов человека с вердиктами судьи (согласованность).
"""
import csv
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
PAIR_DIR = os.path.join(AUK, "local_tests", "pairwise_v1")
AXES = ("text_fidelity", "endings", "naturalness", "prosody", "accent",
        "palatalization", "stress", "artifacts")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main():
    rows = jl(os.path.join(D, "s13_phonetics_results.jsonl"))
    ok = [r for r in rows if r.get("status") == "ok"]
    print("rows:", len(rows), "ok:", len(ok))

    # варианты из id (prNN_side_variant)
    def variant(tid):
        parts = tid.rsplit("_", 1)
        return parts[1] if len(parts) == 2 else "?"

    by_var = defaultdict(list)
    subs = Counter()
    mangled = Counter()
    for r in ok:
        j = r["judge"]
        by_var[variant(r["task_id"])].append(j)
        for s in j.get("phoneme_substitutions", []) or []:
            subs[str(s)] += 1
        for w in j.get("words_mangled", []) or []:
            mangled[str(w).lower()] += 1

    def m(js, k):
        vals = [j.get(k) for j in js if isinstance(j.get(k), int)]
        return round(sum(vals) / len(vals), 2) if vals else None

    L = ["# S13b — произношение под судьёй (80 файлов слепого A/B, группа phonetics)", "",
         "Контекст: пользователь услышал «клучи» вместо «ключи», «ми» вместо «мы», потерю ударений,",
         "местами «китайский» акцент. Прогон подтверждает/опровергает по осям схемы phonetics",
         "(0–10, выше = лучше; accent: 10 = нейтральный русский).", "",
         "## Средние по осям (варианты)", "",
         "| ось | s7 (v1.0) | s8b |", "|---|---|---|"]
    s7, s8 = by_var.get("s7", []), by_var.get("s8", [])
    for k in AXES:
        L.append(f"| {k} | {m(s7,k)} | {m(s8,k)} |")
    L += [f"| overall | {m(s7,'overall')} | {m(s8,'overall')} |", ""]

    # брак по вердиктам
    for name, js in (("s7", s7), ("s8b", s8)):
        vd = Counter(j.get("verdict") for j in js)
        L.append(f"Вердикты {name}: {dict(vd)} (n={len(js)})")
    L.append("")

    # низкие оси
    L += ["## Доля файлов с проблемами по осям (оценка ≤ 5)", "",
          "| ось | s7 | s8b |", "|---|---|---|"]
    for k in ("accent", "palatalization", "stress", "endings", "text_fidelity"):
        f7 = sum(1 for j in s7 if isinstance(j.get(k), int) and j[k] <= 5)
        f8 = sum(1 for j in s8 if isinstance(j.get(k), int) and j[k] <= 5)
        L.append(f"| {k} ≤5 | {f7}/{len(s7)} | {f8}/{len(s8)} |")
    L.append("")

    # фонемные подмены
    L += ["## Фонемные подмены (топ судьи)", ""]
    if subs:
        L.append("| подмена | n |")
        L.append("|---|---|")
        for s, n in subs.most_common(15):
            L.append(f"| {s} | {n} |")
    else:
        L.append("(судья не вернул фонемных подмен)")
    L += ["", "## Прожёванные слова (топ words_mangled)", ""]
    if mangled:
        L.append("| слово | n |")
        L.append("|---|---|")
        for w, n in mangled.most_common(15):
            L.append(f"| {w} | {n} |")
    else:
        L.append("(нет)")
    L.append("")

    # согласованность с human-флагами
    ch = {r["id"]: r for r in csv.DictReader(
        open(os.path.join(PAIR_DIR, "pairwise_v2_choices.csv"), encoding="utf-8"))}
    agree = both = 0
    for r in ok:
        tid = r["task_id"]  # prNN_side_variant
        pid, side = tid.split("_")[0], tid.split("_")[1]
        c = ch.get(pid)
        if not c:
            continue
        hf = str(c.get(f"flag_{side}_mumbled", "0")).strip() == "1"
        jb = r["judge"].get("verdict") == "брак" or (isinstance(r["judge"].get("text_fidelity"), int)
                                                      and r["judge"]["text_fidelity"] <= 5)
        both += 1
        if hf and jb:
            agree += 1
    L += ["## Согласованность человек ↔ судья", "",
          f"Пар (файлов) сравнено: {both}; человек отметил «жуёт» И судья брак/fidelity≤5: {agree}"
          f" ({agree/max(1,both)*100:.0f}%)", "",
          "Интерпретация: низкая согласованность + низкие оси у обоих вариантов =",
          "системная проблема произношения всей ветви (не чекпойнта) — подтверждает",
          "«оба плохи» в 18/40 человеческих пар.", "",
          "## Вывод", "",
          "Заполняется после просмотра таблиц: если accent/palatalization/stress ≤6 у обоих",
          "вариантов — наблюдение пользователя ПОДТВЕРЖДЕНО судьёй; это новый топ-приоритет",
          "(выше S10 first-shot): произносительный датасет/фонетический буткемп (S16)."]

    open(os.path.join(D, "S13_PHONETICS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote S13_PHONETICS.md")
    print("s7 axes:", {k: m(s7, k) for k in AXES})
    print("s8 axes:", {k: m(s8, k) for k in AXES})
    print("top subs:", subs.most_common(6))


if __name__ == "__main__":
    main()
