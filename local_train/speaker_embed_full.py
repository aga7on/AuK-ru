"""Шаг 5, полный проход: эмбеддинги спикеров по ru_mfa_manifest (мультипроцесс, resumable).

Индекс: corpus/corpus_index.jsonl (нормализованные пути). Шарды: corpus/emb/emb_XXXXX.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
sys.path.insert(0, LOCAL)

MANIFEST = r"G:\AI\kyutai-ru\data\ru_mfa_manifest.jsonl"
CORPUS = os.path.join(LOCAL, "corpus")
INDEX = os.path.join(CORPUS, "corpus_index.jsonl")
EMB = os.path.join(CORPUS, "emb")
SHARD = 20000


def build_index():
    if os.path.exists(INDEX):
        n = sum(1 for _ in open(INDEX, encoding="utf-8"))
        print(f"index exists: {n} rows")
        return n
    os.makedirs(CORPUS, exist_ok=True)
    n = 0
    with open(MANIFEST, encoding="utf-8") as fin, open(INDEX, "w", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            p = o.get("path", "").replace("X:\\MediaForge\\kyutai-ru-data\\", "G:\\AI\\kyutai-ru\\data\\")
            fout.write(json.dumps({"p": p, "d": o.get("duration", 0), "t": o.get("transcript", "")},
                                  ensure_ascii=False) + "\n")
            n += 1
    print(f"index built: {n} rows")
    return n


def embed_shard(sid):
    dst = os.path.join(EMB, f"emb_{sid:05d}.npz")
    if os.path.exists(dst):
        return sid, -1, 0.0
    from speaker_embed import embed_path

    i0 = sid * SHARD
    lines = []
    with open(INDEX, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < i0:
                continue
            if i >= i0 + SHARD:
                break
            lines.append(json.loads(line))
    vecs = np.zeros((len(lines), 256), dtype=np.float32)
    ok = np.zeros(len(lines), dtype=bool)
    t0 = time.time()
    for j, o in enumerate(lines):
        try:
            v = embed_path(o["p"])
            vecs[j] = v
            ok[j] = True
        except Exception:
            ok[j] = False
    np.savez_compressed(dst, vecs=vecs, ok=ok, i0=i0)
    return sid, int(ok.sum()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="только первые N клипов (тест)")
    ap.add_argument("--test", type=int, default=0, help="тест-режим: N клипов в одном процессе")
    args = ap.parse_args()

    n = build_index()
    os.makedirs(EMB, exist_ok=True)

    if args.test:
        from speaker_embed import embed_path

        t0 = time.time()
        ok = 0
        vecs = []
        with open(INDEX, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.test:
                    break
                o = json.loads(line)
                try:
                    vecs.append(embed_path(o["p"]))
                    ok += 1
                except Exception:
                    pass
        dt = time.time() - t0
        print(f"TEST: {ok}/{args.test} ok in {dt:.1f}s -> {dt/max(args.test,1)*1000:.0f} ms/clip")
        return

    n_shards = (min(n, args.limit) if args.limit else n + SHARD - 1) // SHARD
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pending = []
    for sid in range(n_shards):
        if not os.path.exists(os.path.join(EMB, f"emb_{sid:05d}.npz")):
            pending.append(sid)
    print(f"shards: total={n_shards}, pending={len(pending)}, workers={args.workers}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(embed_shard, sid): sid for sid in pending}
        for fut in as_completed(futs):
            sid, nok, dt = fut.result()
            done += 1
            if done % 5 == 0 or done == len(pending):
                el = time.time() - t0
                eta = el / done * (len(pending) - done)
                print(f"[{done}/{len(pending)}] shard {sid}: ok={nok} ({dt:.0f}s) "
                      f"ETA {eta/60:.0f} min", flush=True)
    print("EMBED_ALL_DONE")


if __name__ == "__main__":
    main()
