"""Best-of-3 seeds для 25 клонов контроль-пака: меняется ли sim от seed.
Параметризованный аналог s5_seed_probe.py (не хардкодит чекпойнт).
usage: python seed_probe_s7.py --ckpt <merged.safetensors> --config <config.yaml> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio
from speaker_embed import embed_path

pack = json.load(open(r"G:\AI\AuK\local_tests\eval_pack\pack.json", encoding="utf-8"))
clones = [e for e in pack if e.get("kind") == "clone"]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=r"G:\AI\AuK\ckpts\Qwen2.5-Omni-3B",
                   cpu_offload=True, device=args.device, dtype="bf16")
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    res = []
    for e in clones:
        ref_e = embed_path(e["ref"])
        sims = []
        for seed in (7, 123, 999):
            content = [{"type": "text", "text": e["instruction"]},
                       {"type": "audio", "audio": e["ref"]}]
            audio, sr = eng.generate([{"role": "user", "content": content}], audio=e["ref"],
                                     gen_seconds=float(sf.info(e["ref"]).duration) + 0.5,
                                     nfe=64, cfg_strength=2.0, seed=seed)
            p = os.path.join(outdir, e["id"] + f"_s{seed}.wav")
            save_audio(audio, sr, p)
            sims.append(round(cos(ref_e, embed_path(p)), 4))
        res.append({"id": e["id"], "ref": e["ref"], "sims": sims, "best": max(sims)})
        print(e["id"], sims, "best", max(sims), flush=True)

    json.dump(res, open(os.path.join(outdir, "results.json"), "w", encoding="utf-8"), indent=1)
    bests = sorted(r["best"] for r in res)
    median = bests[len(bests) // 2]
    print(f"SEED_PROBE_DONE n={len(res)} best_of3_median={median}")


if __name__ == "__main__":
    main()