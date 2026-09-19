"""Сплиты v2 (честные): по «уверенным» кластерам, неоднозначное — только в train.

- confident cluster: size>=4 и ближайший ДРУГОЙ центроид < NEIGH_TH (0.50).
- dev/final собираются ТОЛЬКО из confident-кластеров (8%/4% клипов);
- все остальные клипы (неуверенные, borderline, некластеризованные) → train.
Выход: corpus/split_manifest.jsonl (i,p,d,t,cluster,split,confident,tags) + split_stats.json
+ corpus/s2_pairs_v2.jsonl (пары клонирования из clusters_v2, train, ≤4/кластер).
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
INDEX = os.path.join(CORPUS, "corpus_index.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-frac", type=float, default=0.08)
    ap.add_argument("--final-frac", type=float, default=0.04)
    ap.add_argument("--neigh-th", type=float, default=0.55)
    ap.add_argument("--min-conf-size", type=int, default=8)
    ap.add_argument("--min-pair-cluster", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import faiss

    labels = np.load(os.path.join(CORPUS, "clusters_v2.npy"))
    cz = np.load(os.path.join(CORPUS, "centroids_v2.npz"))
    means, counts = cz["means"], cz["counts"]
    print(f"clusters: {len(means)}, clips labelled: {(labels>=0).sum()}", flush=True)

    idx = faiss.IndexFlatIP(means.shape[1])
    idx.add(np.ascontiguousarray(means))
    D, I = idx.search(np.ascontiguousarray(means), 2)
    best_other = np.where(I[:, 0] == np.arange(len(means)), D[:, 1], D[:, 0])
    confident = (counts >= args.min_conf_size) & (best_other < args.neigh_th)
    print(f"confident clusters: {int(confident.sum())} of {len(means)} "
          f"(clips {int(counts[confident].sum())})", flush=True)

    rng = random.Random(args.seed)
    conf_ids = [k for k in range(len(means)) if confident[k] and counts[k] <= 2000]
    rng.shuffle(conf_ids)
    total_clips = int(counts[confident].sum())
    n_dev = int(total_clips * args.dev_frac)
    n_fin = int(total_clips * args.final_frac)
    dev_ids, fin_ids = set(), set()
    acc = 0
    for k in conf_ids:
        if acc >= n_dev:
            break
        if acc + int(counts[k]) > 1.5 * n_dev and acc > 0:
            continue
        dev_ids.add(k)
        acc += int(counts[k])
    acc2 = 0
    for k in conf_ids:
        if k in dev_ids:
            continue
        if acc2 >= n_fin:
            break
        if acc2 + int(counts[k]) > 1.5 * n_fin and acc2 > 0:
            continue
        fin_ids.add(k)
        acc2 += int(counts[k])
    print(f"dev clusters: {len(dev_ids)} (clips {acc}), final clusters: {len(fin_ids)} (clips {acc2})")

    import re
    NUM_RE = re.compile(r"\b(\d+|[а-яё]*дцать|[а-яё]*сот|[а-яё]*десят|тысяч[а-яё]*|сто|сорок)\b")
    CLUSTER_RE = re.compile(r"(вз|встр|съ[её]|объ[еёя]|подъ|вспл|вздр|ств|здр|жд|вств)")
    SOFT_RE = re.compile(r"[а-яё]ь\b")

    def tags_of(t):
        t = t.lower()
        out = []
        if NUM_RE.search(t):
            out.append("numerals")
        if CLUSTER_RE.search(t):
            out.append("clusters")
        if SOFT_RE.search(t):
            out.append("soft_sign")
        return out or ["base"]

    stats = defaultdict(int)
    with open(INDEX, encoding="utf-8") as fin, \
            open(os.path.join(CORPUS, "split_manifest.jsonl"), "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            o = json.loads(line)
            cl = int(labels[i]) if i < len(labels) else -1
            if cl in dev_ids:
                split = "dev"
            elif cl in fin_ids:
                split = "final"
            else:
                split = "train"
            conf = bool(confident[cl]) if cl >= 0 else False
            row = {"i": i, "p": o["p"], "d": o["d"], "t": o.get("t", "")[:300],
                   "cluster": cl, "split": split, "confident": conf, "tags": tags_of(o.get("t", ""))}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[f"split_{split}"] += 1

    # ---- пары клонирования из clusters_v2 (train, кластеры >= min_pair_cluster) ----
    by_cluster = defaultdict(list)
    eligible = set()
    for k in range(len(counts)):
        if counts[k] >= args.min_pair_cluster and k not in dev_ids and k not in fin_ids:
            eligible.add(k)
    with open(INDEX, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            o = json.loads(line)
            cl = int(labels[i]) if i < len(labels) else -1
            if cl in eligible and 2.0 <= float(o.get("d", 0)) <= 8.0:
                by_cluster[cl].append({"p": o["p"], "d": o["d"], "t": o.get("t", "")})
    print(f"pair-eligible clusters: {len(by_cluster)}", flush=True)

    rng2 = random.Random(args.seed + 1)
    pairs_lines = []
    for cl, items in by_cluster.items():
        tries = 0
        made_c = 0
        while made_c < 4 and tries < 10 and len(items) >= 2:
            a, b = rng2.sample(items, 2)
            tries += 1
            if a["t"][:40].lower() == b["t"][:40].lower():
                continue
            pairs_lines.append({"ref": a["p"], "ref_d": a["d"], "ref_t": a["t"][:300],
                                "target": b["p"], "target_d": b["d"], "target_t": b["t"][:300],
                                "cluster": cl})
            made_c += 1
    with open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), "w", encoding="utf-8") as f:
        for r in pairs_lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["pairs_v2"] = len(pairs_lines)

    stats.update({
        "confident_clusters": int(confident.sum()),
        "confident_clips": int(counts[confident].sum()),
        "dev_clips": acc, "final_clips": acc2,
        "neigh_th": args.neigh_th, "min_conf_size": args.min_conf_size,
    })
    json.dump(dict(stats), open(os.path.join(CORPUS, "split_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print("SPLIT_V2_DONE")


if __name__ == "__main__":
    main()
