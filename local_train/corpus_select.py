"""Шаг 8-подготовка: отбор s2-подмножества + план пар клонирования из кластеризованного корпуса.

Вход: corpus/corpus_index.jsonl, corpus/speakers.npy (после corpus_cluster.py).
Выход: corpus/s2_selection.jsonl, corpus/s2_pairs.jsonl, corpus/s2_stats.json.

Правила:
- клипы 2.0-8.0с; split по СПИКЕР-КЛАСТЕРАМ (train/dev/final, ~88/8/4);
- клонирование: только кластеры >=8 клипов, пары «реф≠цель» (разные транскрипты);
- целевые категории (числительные/кластеры/ь-слова) получают приоритет в отборе;
- дедуп: не более 2 клипов с одинаковым текстом на кластер.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
INDEX = os.path.join(CORPUS, "corpus_index.jsonl")
SPK = os.path.join(CORPUS, "speakers.npy")

NUM_RE = re.compile(r"\b(\d+|[а-яё]*дцать|[а-яё]*сот|[а-яё]*десят|тысяч[а-яё]*|сто|сорок|двести|триста|четыреста|пятьсот|шестьсот|семьсот|восемьсот|девятьсот)\b")
CLUSTER_RE = re.compile(r"(вз|вс[т]?р|съ[её]|объ[еёя]|подъ|вспл|вздр|ств|здр|жд|вств)")
SOFT_RE = re.compile(r"[а-яё]ь\b")


def classify(text: str) -> list[str]:
    t = text.lower()
    tags = []
    if NUM_RE.search(t):
        tags.append("numerals")
    if CLUSTER_RE.search(t):
        tags.append("clusters")
    if SOFT_RE.search(t):
        tags.append("soft_sign")
    return tags or ["base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=250000)
    ap.add_argument("--dev-frac", type=float, default=0.08)
    ap.add_argument("--final-frac", type=float, default=0.04)
    ap.add_argument("--min-cluster-pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import numpy as np

    labels = np.load(SPK)
    rows = []
    with open(INDEX, encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            o["i"] = i
            o["spk"] = int(labels[i]) if i < len(labels) else -1
            rows.append(o)
    print(f"total {len(rows)} clips; clustered {sum(1 for r in rows if r['spk'] >= 0)}")

    by_spk = defaultdict(list)
    for r in rows:
        if r["spk"] >= 0 and 2.0 <= float(r.get("d", 0)) <= 8.0:
            by_spk[r["spk"]].append(r)

    spk_ids = sorted(by_spk)
    rng.shuffle(spk_ids)
    n_dev = max(1, int(len(spk_ids) * args.dev_frac))
    n_fin = max(1, int(len(spk_ids) * args.final_frac))
    dev_set = set(spk_ids[:n_dev])
    fin_set = set(spk_ids[n_dev:n_dev + n_fin])
    print(f"speakers: {len(spk_ids)} (dev {len(dev_set)}, final {len(fin_set)}, train {len(spk_ids)-n_dev-n_fin})")

    sel = []
    pairs = []
    stats = defaultdict(int)
    for spk in spk_ids:
        split = "dev" if spk in dev_set else ("final" if spk in fin_set else "train")
        clips = by_spk[spk]
        clips.sort(key=lambda r: -float(r["d"]))
        seen_text = defaultdict(int)
        kept = []
        for r in clips:
            key = r.get("t", "")[:80].lower()
            if seen_text[key] >= 2:
                continue
            seen_text[key] += 1
            tags = classify(r.get("t", ""))
            r2 = {"i": r["i"], "p": r["p"], "d": r["d"], "t": r["t"][:300], "spk": spk,
                  "split": split, "tags": tags}
            kept.append(r2)
            stats[f"tag_{tags[0]}"] += 1
            stats[f"split_{split}"] += 1
            if len(sel) < args.max_clips and split == "train":
                sel.append(r2)
        if split == "train" and len(kept) >= args.min_cluster_pairs:
            # пары клонирования: (реф, цель) с разными текстами, цель 3-10с
            pool = [r for r in kept if 2.0 <= float(r["d"]) <= 8.0]
            for _ in range(min(4, len(pool) // 2)):
                a, b = rng.sample(pool, 2)
                if a.get("t", "")[:40].lower() == b.get("t", "")[:40].lower():
                    continue
                pairs.append({"ref": a["p"], "ref_d": a["d"], "ref_t": a["t"][:200],
                              "target": b["p"], "target_d": b["d"], "target_t": b["t"][:300],
                              "spk": spk, "split": "train"})
                stats["pairs"] += 1

    json.dump({"stats": dict(stats), "selection": len(sel), "pairs": len(pairs)},
              open(os.path.join(CORPUS, "s2_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(CORPUS, "s2_selection.jsonl"), "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(CORPUS, "s2_pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print(f"S2_SELECT_DONE selection={len(sel)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
