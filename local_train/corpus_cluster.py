"""Шаг 5, глобальная кластеризация: leader-follower (cosine) + running-mean центроиды.

Индекс faiss пересобирается каждые REBUILD назначений (Flat до 8k центроидов, дальше IVFFlat).
Консервативный порог (0.55) — «не сливать разных людей».
"""
import argparse
import json
import os
import sys

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
EMB = os.path.join(CORPUS, "emb")
REBUILD = 100000


def make_index(means, faiss):
    arr = np.ascontiguousarray(np.stack(means), dtype=np.float32)
    if len(means) < 8192:
        idx = faiss.IndexFlatIP(arr.shape[1])
        idx.add(arr)
        return idx
    nlist = 1024
    quantizer = faiss.IndexFlatIP(arr.shape[1])
    idx = faiss.IndexIVFFlat(quantizer, arr.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    train = arr[np.random.default_rng(0).choice(len(arr), size=min(len(arr), 65536), replace=False)]
    idx.train(np.ascontiguousarray(train))
    idx.add(arr)
    idx.nprobe = 16
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=0.55)
    args = ap.parse_args()
    import faiss

    shards = sorted(f for f in os.listdir(EMB) if f.startswith("emb_"))
    n_total = 0
    for s in shards:
        n_total += int(np.load(os.path.join(EMB, s))["ok"].sum())
    print(f"shards={len(shards)} valid_clips={n_total}", flush=True)

    labels = {}
    means = []
    counts = []
    index = None
    processed = 0
    next_rebuild = REBUILD
    new_since = 0
    rng = np.random.default_rng(0)

    for s in shards:
        z = np.load(os.path.join(EMB, s))
        vecs, ok, i0 = z["vecs"], z["ok"], int(z["i0"])
        pos = i0 + np.flatnonzero(ok)
        valid = vecs[ok]
        for j in range(len(valid)):
            v = valid[j]
            if index is None or processed >= next_rebuild or new_since >= 8192:
                index = make_index(means, faiss) if means else None
                next_rebuild = processed + REBUILD
                new_since = 0
            c = -1
            if index is not None and len(means) > 0:
                D, I = index.search(v[None, :], 1)
                if D[0, 0] >= args.th:
                    c = int(I[0, 0])
            if c < 0:
                means.append(v.copy())
                counts.append(1)
                c = len(means) - 1
                new_since += 1
            else:
                counts[c] += 1
                means[c] = means[c] + (v - means[c]) / counts[c]
                m = means[c]
                nrm = np.linalg.norm(m)
                if nrm > 1e-6:
                    means[c] = m / nrm
            labels[int(pos[j])] = c
            processed += 1
        print(f"  shard {s}: processed={processed} centroids={len(means)}", flush=True)

    n_arr = max(labels.keys()) + 1
    lab = np.full(n_arr, -1, dtype=np.int64)
    for k, v in labels.items():
        lab[k] = v
    np.save(os.path.join(CORPUS, "speakers.npy"), lab)
    sizes = np.bincount(lab[lab >= 0])
    big = sizes[sizes >= 8]
    stats = {"clips": int((lab >= 0).sum()), "centroids": int(len(sizes)),
             "clusters_ge8": int(len(big)), "clips_in_ge8": int(big.sum()),
             "top_sizes": sorted(sizes.tolist(), reverse=True)[:15], "threshold": args.th}
    json.dump(stats, open(os.path.join(CORPUS, "speakers_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("CLUSTER_DONE")


if __name__ == "__main__":
    main()
