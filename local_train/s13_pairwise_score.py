# -*- coding: utf-8 -*-
"""S13: подсчёт результатов human pairwise A/B.

Вход: exports/pairwise_v1_choices.csv (экспорт из listen.html) +
local_tests/pairwise_v1/SECRET_pair_map.csv (маппинг A/B→вариант; ОТКРЫВАЕТСЯ только здесь,
после завершения прослушивания — слепой протокол).

Выход: local_train/reports/deepseek_supervised/S13_HUMAN_AB.md — win-rate по вариантам,
по категориям (tts_hard/clone/emotion) и по тексту (худшие расхождения).

usage: python s13_pairwise_score.py --choices <csv>
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
PAIR_DIR = os.path.join(AUK, "local_tests", "pairwise_v1")
SECRET = os.path.join(PAIR_DIR, "SECRET_pair_map.csv")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "S13_HUMAN_AB.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", required=True,
                    help="CSV из listen.html v2 (id,kind,choice,flag_a_mumbled,flag_b_mumbled)")
    args = ap.parse_args()

    secret = {}
    with open(SECRET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            secret[r["id"]] = r  # a_variant / b_variant

    rows = []
    with open(args.choices, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def flag(r, k):
        return str(r.get(k, "")).strip() in ("1", "true", "True")

    total = len(rows)
    answered = [r for r in rows if (r.get("choice") or "").strip().upper() in ("A", "B")]
    both_bad = [r for r in rows if (r.get("choice") or "").strip().upper() == "X"]
    by_variant = Counter()
    by_kind = defaultdict(Counter)
    mumble = Counter()   # вариант → число флагов «жуёт слова»
    mumble_n = Counter()  # вариант → число оценённых пар (для доли)
    both_flagged = 0
    for r in rows:
        pid = r["id"]
        s = secret.get(pid)
        if not s:
            continue
        fa, fb = flag(r, "flag_a_mumbled"), flag(r, "flag_b_mumbled")
        if fa and fb:
            both_flagged += 1
        if fa or fb:
            mumble_n[s["a_variant"]] += 1 if fa else 0
            mumble_n[s["b_variant"]] += 1 if fb else 0
            if fa:
                mumble[s["a_variant"]] += 1
            if fb:
                mumble[s["b_variant"]] += 1
    for r in answered:
        pid, ch = r["id"], r["choice"].strip().upper()
        s = secret.get(pid)
        if not s:
            continue
        variant = s["a_variant"] if ch == "A" else s["b_variant"]
        by_variant[variant] += 1
        by_kind[r.get("kind", "?")][variant] += 1

    n_with_choice = len(answered) + len(both_bad)
    L = ["# S13 HUMAN A/B — pairwise прослушивание v2 (v1.0 s7@5750 vs s8b@8500)", "",
         f"Пар: {total}; с выбором A/B: {len(answered)}; «оба плохи»: {len(both_bad)}; без ответа: {total - n_with_choice}", "",
         "## Итог по вариантам (только пары с выбором A/B)", "", "| вариант | побед | доля |", "|---|---|---|"]
    for v, n in by_variant.most_common():
        L.append(f"| {v} | {n} | {n/max(1,len(answered))*100:.0f}% |")
    L += ["", "## Флаги «жуёт слова» (v2)", "",
          "| вариант | флагов | доля пар с флагом |", "|---|---|---|"]
    for v in sorted(mumble_n or mumble):
        n = mumble.get(v, 0)
        L.append(f"| {v} | {n} | {n/max(1,total)*100:.0f}% от {total} пар |")
    L += ["", f"Пар, где «жуёт» у ОБЕИХ вариантов: **{both_flagged}** — это верхняя оценка общей",
          "проблемы дикции (не различает модели); «оба плохи»: " + str(len(both_bad)) + ".", "",
          "## По категориям (выборы A/B)", "", "| категория | s7 | s8b |", "|---|---|---|"]
    for k, c in sorted(by_kind.items()):
        L.append(f"| {k} | {c.get('s7',0)} | {c.get('s8b',0)} |")
    L += ["", "Слепой протокол: маппинг A/B раскрыт только здесь (SECRET_pair_map.csv).",
          "Решение по канону — с учётом этого результата (ROADMAP S13).",
          "Интерпретация флагов: если «жуёт» у обоих вариантов в >50% пар — проблема доменная",
          "(тексты/рефы), а не конкретного чекпойнта; перекос в одну сторону — сигнал для микса.",
          "Флаги также идут в калибровку composite-reranker (S11): «жуёт» ≈ words_mangled судьи."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("answered:", len(answered), "both_bad:", len(both_bad),
          "by_variant:", dict(by_variant), "mumble:", dict(mumble), "both_flagged:", both_flagged)


if __name__ == "__main__":
    main()
