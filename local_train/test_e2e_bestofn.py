"""End-to-end test of run_generate with best-of-2 + trim (GPU1, needs hold on controller)."""
import os
import sys

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")

import auk.infer.infer_gradio as ig  # noqa: E402
from auk.infer.infer_auk import AukInfer  # noqa: E402

AUK = r"G:\AI\AuK"
MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_13400.safetensors")
REF = os.path.join(AUK, "local_tests", "user_ref.wav")
OUT = os.path.join(AUK, "local_tests", "e2e_bestofn.wav")
TEXT = "Позвони мне, пожалуйста, когда освободишься."
INSTR = f"Say the following with the same voice: '{TEXT}'"

ig.CKPT_PATHS["test"] = MERGED
ig.ENGINES["test"] = AukInfer(
    config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
    ckpt_path=MERGED,
    qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
    cpu_offload=True,
    device="cuda:1",
    dtype="bf16",
)

sr, pcm = ig.run_generate(
    "test", REF, INSTR, 6.0, "", "", 64, 2.0, 1234,
    translit=False, accentize=True, bestofn=2, trim=True,
)
sf.write(OUT, pcm, sr, subtype="PCM_16")
print(f"saved {OUT}: {len(pcm)/sr:.2f}s @ {sr}")

from auk.infer import quality  # noqa: E402

import torch  # noqa: E402

x, srx = sf.read(OUT, dtype="float32", always_2d=True)
heard = quality.transcribe(torch.from_numpy(x.T.copy()), srx)
print("gigaam heard:", heard)
print("recall:", quality.recall(TEXT, heard))
print("E2E_OK")
