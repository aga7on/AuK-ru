"""Best-of-3 seeds для 25 клонов контроль-пака на s5@4500: меняется ли sim от seed."""
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


eng = AukInfer(config_path=r"G:\AI\AuK\local_train\run_s5\merged\config.yaml",
               ckpt_path=r"G:\AI\AuK\local_train\run_s5\merged\auk_s5_4500.safetensors",
               qwen_path=r"G:\AI\AuK\ckpts\Qwen2.5-Omni-3B",
               cpu_offload=True, device="cuda:1", dtype="bf16")
outdir = r"G:\AI\AuK\local_tests\tmp_seed_probe"
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
    res.append((e["id"], sims))
    print(e["id"], sims, "best", max(sims), flush=True)

import statistics as st
bests = [max(r[1]) for r in res]
firsts = [r[1][0] for r in res]
print("seed7 median", round(st.median(firsts), 4))
print("BEST-OF-3 median", round(st.median(bests), 4), "mean", round(st.mean(bests), 4))
json.dump({"per_clone": [(r[0], r[1]) for r in res]},
          open(os.path.join(outdir, "seed_probe.json"), "w", encoding="utf-8"), indent=1)
