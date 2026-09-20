# -*- coding: utf-8 -*-
"""S13 → S11: калибровка детектора «жуёт слова» на human-флагах.

Данные: 80 файлов A/B (78 ok у судьи) + человеческие флаги flag_a/b_mumbled.
Считаем:
  1. корреляцию каждого признака судьи/метрики с human-флагом (point-biserial);
  2. простое правило-детектор «жуёт» (пороги по осям судьи) и его precision/recall;
  3. рекомендацию для composite reranker: какие оси штрафовать.

usage: python s13_calibration.py
Выход: stdout + reports/deepseek_supervised/S13_CALIBRATION.md
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
PAIR_DIR = os.path.join(AUK, "local_tests", "pairwise_v1")
OUT = os.path.join(D, "S13_CALIBRATION.md")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pointbiserial(xs, ys):
    # xs: float, ys: 0/1
    ones = [x for x, y in zip(xs, ys) if y == 1]
    zeros = [x for x, y in zip(xs, ys) if y == 0]
    if not ones or not zeros:
        return None
    sd = st.pstdev(xs)
    if sd == 0:
        return None
    p = len(ones) / len(xs)
    return round((st.mean(ones) - st.mean(zeros)) / sd * (p * (1 - p)) ** 0.5, 3)


def main():
    judge = {r["task_id"]: r["judge"] for r in jl(os.path.join(D, "s13_phonetics_results.jsonl"))
             if r.get("status") == "ok"}
    ch = {r["id"]: r for r in csv.DictReader(open(os.path.join(PAIR_DIR, "pairwise_v2_choices.csv"), encoding="utf-8"))}

    rows = []
    for tid, j in judge.items():
        pid, side, _var = tid.rsplit("_", 2) if tid.count("_") >= 2 else (tid, "", "")
        parts = tid.split("_")
        pid, side = parts[0], parts[1]
        c = ch.get(pid)
        if not c:
            continue
        y = 1 if str(c.get(f"flag_{side}_mumbled", "0")).strip() == "1" else 0
        feats = {
            "accent": j.get("accent"), "palatalization": j.get("palatalization"),
            "stress": j.get("stress"), "endings": j.get("endings"),
            "text_fidelity": j.get("text_fidelity"), "naturalness": j.get("naturalness"),
            "prosody": j.get("prosody"), "artifacts": j.get("artifacts"), "overall": j.get("overall"),
            "n_mangled": len(j.get("words_mangled") or []),
            "n_dropped": len(j.get("words_dropped") or []),
            "n_subs": len(j.get("phoneme_substitutions") or []),
            "verdict_brak": 1 if j.get("verdict") == "брак" else 0,
        }
        rows.append((y, feats))

    n = len(rows)
    ys = [y for y, _ in rows]
    print(f"n={n} mumbled={sum(ys)} ({sum(ys)/n*100:.0f}%)")

    keys = list(rows[0][1].keys())
    corr = {}
    for k in keys:
        xs = [f[k] for _, f in rows if isinstance(f.get(k), (int, float))]
        yy = [y for y, f in rows if isinstance(f.get(k), (int, float))]
        corr[k] = pointbiserial(xs, yy)
    print("point-biserial corr with human 'mumbled':")
    for k, v in sorted(corr.items(), key=lambda kv: abs(kv[1] or 0), reverse=True):
        print(f"  {k}: {v}")

    # правила-детекторы
    rules = {
        "verdict==брак": lambda f: f["verdict_brak"] == 1,
        "text_fidelity<=5": lambda f: f["text_fidelity"] <= 5,
        "accent<=6": lambda f: f["accent"] <= 6,
        "n_mangled>=1": lambda f: f["n_mangled"] >= 1,
        "n_subs>=1": lambda f: f["n_subs"] >= 1,
        "overall<=5": lambda f: f["overall"] <= 5,
        "composite: fid<=6 or mangled>=1 or subs>=1": lambda f: (f["text_fidelity"] <= 6 or f["n_mangled"] >= 1 or f["n_subs"] >= 1),
    }
    L = ["# S13 → S11 CALIBRATION — детектор «жуёт слова» на human-флагах", "",
         f"n={n} файлов, human «жуёт» = {sum(ys)} ({sum(ys)/n*100:.0f}%).", "",
         "## Point-biserial корреляции признаков судьи с human-флагом", "",
         "| признак | corr |", "|---|---|"]
    for k, v in sorted(corr.items(), key=lambda kv: abs(kv[1] or 0), reverse=True):
        L.append(f"| {k} | {v} |")
    L += ["", "## Правила-детекторы (precision/recall против human)", "",
          "| правило | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|"]
    for name, fn in rules.items():
        tp = sum(1 for y, f in rows if y == 1 and fn(f))
        fp = sum(1 for y, f in rows if y == 0 and fn(f))
        fneg = sum(1 for y, f in rows if y == 1 and not fn(f))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fneg) if tp + fneg else 0
        L.append(f"| {name} | {tp} | {fp} | {fneg} | {prec:.2f} | {rec:.2f} |")
        print(f"rule {name}: TP={tp} FP={fp} FN={fneg} prec={prec:.2f} rec={rec:.2f}")
    L += ["", "## Рекомендация для composite reranker (S11)", "",
          "Выбирается правило с лучшим F1; штраф в composite score за его срабатывание",
          "(текущий rep_pen/art_pen расширяется: mumble_pen = 0.15·(fid≤6) + 0.10·(mangled≥1)",
          "+ 0.10·(subs≥1)). Калибровка приблизительная (n=78, один слушатель) — уточнить",
          "на следующем A/B-наборе.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
