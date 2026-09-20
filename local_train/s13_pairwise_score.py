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
    ap.add_argument("--choices", required=True, help="CSV из listen.html (id,kind,choice,...)")
    args = ap.parse_args()

    secret = {}
    with open(SECRET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            secret[r["id"]] = r  # a_variant / b_variant

    rows = []
    with open(args.choices, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    total = len(rows)
    answered = [r for r in rows if (r.get("choice") or "").strip() in ("A", "B")]
    by_variant = Counter()
    by_kind = defaultdict(Counter)
    ties = 0
    for r in answered:
        pid, ch = r["id"], r["choice"].strip().upper()
        s = secret.get(pid)
        if not s:
            continue
        variant = s["a_variant"] if ch == "A" else s["b_variant"]
        by_variant[variant] += 1
        by_kind[r.get("kind", "?")][variant] += 1

    L = ["# S13 HUMAN A/B — pairwise прослушивание (v1.0 s7@5750 vs s8b@8500)", "",
         f"Пар: {total}, отвечено: {len(answered)}, без ответа: {total - len(answered)}", "",
         "## Итог по вариантам", "", "| вариант | побед | доля |", "|---|---|---|"]
    for v, n in by_variant.most_common():
        L.append(f"| {v} | {n} | {n/max(1,len(answered))*100:.0f}% |")
    L += ["", "## По категориям", "", "| категория | s7 | s8b |", "|---|---|---|"]
    for k, c in sorted(by_kind.items()):
        L.append(f"| {k} | {c.get('s7',0)} | {c.get('s8b',0)} |")
    L += ["", "Слепой протокол: маппинг A/B раскрыт только здесь (SECRET_pair_map.csv).",
          "Решение по канону принимается с учётом этого результата (ROADMAP S13)."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("answered:", len(answered), "by_variant:", dict(by_variant))


if __name__ == "__main__":
    main()
