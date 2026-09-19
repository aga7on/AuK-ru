"""П.3: лист проверки идентичности пар — 20 уверенных + 20 пограничных.
Проверяющий слушает ref + target и говорит: «один человек» или «разные».
"""
import json
import os
import random
import sys

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
OUT = os.path.join(LOCAL, "tests_packs", "identity_check")
sys.path.insert(0, LOCAL)
sys.path.insert(0, r"G:\AI\AuK\src")

from speaker_embed import embed_path


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(42)

    pairs = []
    with open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"pairs loaded: {len(pairs)}", flush=True)

    rng.shuffle(pairs)
    confident = []
    borderline = []
    for pr in pairs:
        if len(confident) >= 20 and len(borderline) >= 20:
            break
        if not (os.path.exists(pr["ref"]) and os.path.exists(pr["target"])):
            continue
        try:
            er = embed_path(pr["ref"])
            et = embed_path(pr["target"])
            sim = float(np.dot(er, et))
        except Exception:
            continue
        row = {"ref": pr["ref"], "ref_t": pr["ref_t"], "target": pr["target"],
               "target_t": pr["target_t"], "sim": round(sim, 3), "cluster": pr["cluster"]}
        if sim > 0.7 and len(confident) < 20:
            confident.append(row)
        elif 0.55 <= sim <= 0.6 and len(borderline) < 20:
            borderline.append(row)
        if len(confident) >= 20 and len(borderline) >= 20:
            break
    print(f"confident: {len(confident)}, borderline: {len(borderline)}", flush=True)

    manifest = []
    for i, r in enumerate(confident + borderline):
        bucket = "conf" if i < len(confident) else "border"
        ref_dst = os.path.join(OUT, f"id{bucket}_i{i:02d}_ref.wav")
        tgt_dst = os.path.join(OUT, f"id{bucket}_i{i:02d}_tgt.wav")
        import shutil
        shutil.copy(r["ref"], ref_dst)
        shutil.copy(r["target"], tgt_dst)
        manifest.append({
            "pair_id": f"{bucket}_{i:02d}", "sim": r["sim"], "cluster": r["cluster"],
            "ref_file": ref_dst, "ref_text": r["ref_t"][:100],
            "tgt_file": tgt_dst, "tgt_text": r["target_t"][:100],
            "question": "Один человек говорит в обоих файлах? [да/нет/не уверен]",
        })

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"saved {len(manifest)} pair-check items to {OUT}")
    print("IDENTITY_PACK_DONE")


if __name__ == "__main__":
    main()
