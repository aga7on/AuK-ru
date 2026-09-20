# -*- coding: utf-8 -*-
"""Валидация mumble_penalty (ASR-прокси) против human-флагов «жуёт слова» на 80 A/B файлах.

Если прокси ловит ≥70% human-флагов при FP ≤40% — внедрение в composite reranker оправдано.
usage: python validate_mumble_proxy.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
PAIR_DIR = os.path.join(AUK, "local_tests", "pairwise_v1")

from rerank_composite import mumble_penalty  # noqa: E402
from ru_metrics import transcribe_path  # noqa: E402


def main():
    manifest = {e["id"]: e for e in json.load(open(os.path.join(D, "s13_phonetics_manifest.json"), encoding="utf-8"))}
    ch = {r["id"]: r for r in csv.DictReader(open(os.path.join(PAIR_DIR, "pairwise_v2_choices.csv"), encoding="utf-8"))}

    tp = fp = fn = tn = 0
    errs = []
    for tid, e in manifest.items():
        parts = tid.split("_")
        pid, side = parts[0], parts[1]
        c = ch.get(pid)
        if not c:
            continue
        y = 1 if str(c.get(f"flag_{side}_mumbled", "0")).strip() == "1" else 0
        if not os.path.exists(e["file"]):
            continue
        heard, err = transcribe_path(e["file"])
        if err:
            errs.append(tid)
            continue
        pen = mumble_penalty(e["text"], heard)
        pred = 1 if pen >= 0.15 else 0
        if y and pred: tp += 1
        elif y and not pred: fn += 1
        elif pred: fp += 1
        else: tn += 1

    n = tp + fp + fn + tn
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    print(f"n={n} errors={len(errs)}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} precision={prec:.2f} recall={rec:.2f}")
    print("VERDICT:", "PROXY OK (recall>=0.70, FP<=40%)" if rec >= 0.70 and fp <= 0.40 * n else
          ("MARGINAL" if rec >= 0.55 else "WEAK — прокси не ловит human-флаги, штраф держать малым"))


if __name__ == "__main__":
    main()
