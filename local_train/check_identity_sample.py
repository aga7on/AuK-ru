"""Проверка причин отказов identity на выборке пар (по ревью п.2).

Для каждой отбракованной пары: embed(ref), embed(target), sim, cluster size, transкрипт.
"""
import json
import os
import sys
import random

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
sys.path.insert(0, LOCAL)
sys.path.insert(0, r"G:\AI\AuK\src")

from speaker_embed import embed_path


def main():
    rng = random.Random(3)
    pairs = []
    with open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    rng.shuffle(pairs)
    print(f"total pairs: {len(pairs)}")

    # проверить 30 случайных
    sample = rng.sample(pairs, 30)
    results = []
    for k, pr in enumerate(sample):
        if not (os.path.exists(pr["ref"]) and os.path.exists(pr["target"])):
            continue
        er = embed_path(pr["ref"])
        et = embed_path(pr["target"])
        sim = float(np.dot(er, et))
        results.append({
            "cluster": pr["cluster"], "sim": round(sim, 4),
            "ref_d": round(float(pr["ref_d"]), 2), "target_d": round(float(pr["target_d"]), 2),
            "ref": os.path.basename(pr["ref"]), "target": os.path.basename(pr["target"]),
            "ref_t": pr["ref_t"][:60], "target_t": pr["target_t"][:60],
        })
        print(f"  {k+1:2d}/30 cluster={pr['cluster']:<6} sim={sim:.3f} "
              f"{'PASS' if sim >= 0.6 else 'FAIL'} | ref={pr['ref_d']:.1f}s tgt={pr['target_d']:.1f}s", flush=True)

    sims = [r["sim"] for r in results]
    passes = sum(1 for s in sims if s >= 0.6)
    print(f"\npass rate: {passes}/{len(results)} = {100*passes/max(len(results),1):.0f}%")
    print(f"sim distribution: min={min(sims):.3f} max={max(sims):.3f} "
          f"mean={np.mean(sims):.3f} median={np.median(sims):.3f}")
    print(f"0.5-0.6 band: {sum(1 for s in sims if 0.5 <= s < 0.6)}")
    print(f"0.4-0.5 band: {sum(1 for s in sims if 0.4 <= s < 0.5)}")
    print(f"<0.4: {sum(1 for s in sims if s < 0.4)}")
    json.dump(results, open(r"G:\AI\AuK\local_train\corpus\identity_sample.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
