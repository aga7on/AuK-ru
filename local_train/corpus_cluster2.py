"""Кластеризация спикеров v2: без устаревших центроидов.

- Pass 1: leader-follower, лидеры добавляются в faiss FlatIP сразу; centered-mean пересобирается
  каждые REBUILD назначений (устаревание обновлённых центроидов ограничено REBUILD).
- Pass 2: пересчёт финальных центроидов и переназначение ВСЕХ клипов (refinement).
- Groups: union-find по kNN центроидов (sim>=0.50) — группы для БЕЗОПАСНОГО сплита.
- Отчёты: распределение, borderline-доля (best-second<0.05), проверка 4 рефов.
"""
import argparse
import json
import os
import sys

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
EMB = os.path.join(CORPUS, "emb")
REBUILD = 25000


def load_all():
    shards = sorted(f for f in os.listdir(EMB) if f.startswith("emb_"))
    vecs = []
    kinds = []
    total = 0
    for s in shards:
        z = np.load(os.path.join(EMB, s))
        vecs.append(z["vecs"])
        kinds.append(z["ok"])
        total += len(z["ok"])
    V = np.vstack(vecs)
    OK = np.concatenate(kinds)
    return V, OK


def make_flat(means, faiss):
    arr = np.ascontiguousarray(np.stack(means), dtype=np.float32)
    idx = faiss.IndexFlatIP(arr.shape[1])
    idx.add(arr)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=0.55)
    ap.add_argument("--group-th", type=float, default=0.50)
    args = ap.parse_args()
    import faiss

    V, OK = load_all()
    n = len(V)
    print(f"loaded {n} vectors, ok={int(OK.sum())}", flush=True)

    means = []
    counts = []
    labels = np.full(n, -1, dtype=np.int32)
    index = None
    processed = 0
    next_rebuild = REBUILD

    for i in range(n):
        if not OK[i]:
            continue
        v = V[i]
        if index is None:
            index = make_flat(means, faiss) if means else None
        c = -1
        if index is not None:
            D, I = index.search(v[None, :], 1)
            if D[0, 0] >= args.th:
                c = int(I[0, 0])
        if c < 0:
            means.append(v.copy())
            counts.append(1)
            c = len(means) - 1
            if index is not None:
                index.add(np.ascontiguousarray(v[None, :]))
        else:
            counts[c] += 1
            means[c] = means[c] + (v - means[c]) / counts[c]
            m = means[c]
            nrm = np.linalg.norm(m)
            if nrm > 1e-6:
                means[c] = m / nrm
        labels[i] = c
        processed += 1
        if processed >= next_rebuild:
            index = make_flat(means, faiss) if means else None
            next_rebuild = processed + REBUILD
    print(f"pass1 done: centroids={len(means)}", flush=True)

    # pass 2: refinement
    means2 = []
    counts2 = []
    for lab in range(len(means)):
        sel = np.flatnonzero(labels[OK] == lab) if False else None
    order = np.flatnonzero(OK)
    lab_ok = labels[order]
    K = len(means)
    sums = np.zeros((K, 256), dtype=np.float64)
    cnts = np.zeros(K, dtype=np.int64)
    np.add.at(sums, lab_ok, V[order])
    np.add.at(cnts, lab_ok, 1)
    means2 = []
    counts2 = []
    remap = np.full(K, -1, dtype=np.int32)
    for k in range(K):
        if cnts[k] == 0:
            continue
        m = (sums[k] / cnts[k]).astype(np.float32)
        nrm = np.linalg.norm(m)
        if nrm > 1e-6:
            m = m / nrm
        remap[k] = len(means2)
        means2.append(m)
        counts2.append(int(cnts[k]))
    labels2 = remap[labels]
    labels2[~OK] = -1
    print(f"means recomputed: {len(means2)}", flush=True)

    idx2 = make_flat(means2, faiss)
    borderline = 0
    assigned = 0
    for i in order:
        v = V[i]
        D, I = idx2.search(v[None, :], 2)
        best, second = float(D[0, 0]), float(D[0, 1]) if D.shape[1] > 1 else 0.0
        if best >= args.th:
            labels2[i] = int(I[0, 0])
            assigned += 1
            if best - second < 0.05:
                borderline += 1
    print(f"pass2 done: assigned={assigned} borderline={borderline} "
          f"({100*borderline/max(assigned,1):.1f}%)", flush=True)

    # groups via union-find over centroid kNN
    arr = np.stack(means2)
    idxc = faiss.IndexFlatIP(arr.shape[1])
    idxc.add(np.ascontiguousarray(arr))
    Dc, Ic = idxc.search(arr, min(6, len(means2)))
    parent = list(range(len(means2)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for k in range(len(means2)):
        for j in range(1, Dc.shape[1]):
            if Dc[k, j] >= args.group_th:
                union(k, int(Ic[k, j]))
    roots = {}
    gid = np.full(len(means2), -1, dtype=np.int32)
    for k in range(len(means2)):
        r = find(k)
        if r not in roots:
            roots[r] = len(roots)
        gid[k] = roots[r]
    groups = np.full(n, -1, dtype=np.int32)
    mask = labels2 >= 0
    groups[mask] = gid[labels2[mask]]
    print(f"groups: {len(roots)}", flush=True)

    # ref sanity check
    try:
        sys.path.insert(0, LOCAL)
        from speaker_embed import embed_path
        refs = {
            "user_male": r"G:\AI\AuK\local_tests\user_ref.wav",
            "nat_female": r"G:\AI\AuK\local_tests\ref_ru.wav",
            "alt_male": r"G:\AI\AuK\local_tests\ref_male2.wav",
            "alt_female": r"G:\AI\AuK\local_tests\ref_female2.wav",
        }
        for name, p in refs.items():
            e = embed_path(p)
            D, I = idx2.search(e[None, :], 1)
            print(f"ref {name}: cluster={int(I[0,0])} sim={float(D[0,0]):.3f}")
    except Exception as exc:
        print("ref check skipped:", exc)

    sizes = np.bincount(labels2[labels2 >= 0])
    gsizes = np.bincount(groups[groups >= 0])
    stats = {
        "clips": int((labels2 >= 0).sum()), "clusters": int(len(sizes)),
        "clusters_ge8": int((sizes >= 8).sum()), "clips_in_ge8": int(sizes[sizes >= 8].sum()),
        "groups": int(len(gsizes)), "groups_ge8": int((gsizes >= 8).sum()),
        "clips_in_groups_ge8": int(gsizes[gsizes >= 8].sum()),
        "borderline_pct": round(100 * borderline / max(assigned, 1), 2),
        "top_cluster_sizes": sorted(sizes.tolist(), reverse=True)[:10],
        "top_group_sizes": sorted(gsizes.tolist(), reverse=True)[:10],
        "threshold": args.th, "group_threshold": args.group_th,
    }
    np.save(os.path.join(CORPUS, "clusters_v2.npy"), labels2)
    np.save(os.path.join(CORPUS, "groups_v2.npy"), groups)
    np.savez_compressed(os.path.join(CORPUS, "centroids_v2.npz"),
                        means=np.stack(means2), counts=np.array(counts2, dtype=np.int64))
    json.dump(stats, open(os.path.join(CORPUS, "cluster2_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("CLUSTER2_DONE")


if __name__ == "__main__":
    main()
