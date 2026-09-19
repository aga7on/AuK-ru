"""Smoke test for the best-of-N screening + trim integration (CPU only)."""
import importlib
import sys

import soundfile as sf
import torch

sys.path.insert(0, r"G:\AI\AuK\src")

from auk.infer import quality  # noqa: E402


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(x.T.copy()), sr


D = r"G:\AI\AuK\local_tests\bestofn\cand"
cands = [load(f"{D}\\c04_s1234.wav"), load(f"{D}\\c04_s7.wav"), load(f"{D}\\c04_s42.wav")]
idx = quality.pick_best(cands, "Сельдь под шубой — традиционное новогоднее блюдо.")
print("pick c04 ->", idx, "(expect 0 = s1234)")

src = r"G:\AI\AuK\local_train\run_ru_s1\samples\update_15250_gen.wav"
wav, sr = load(src)
trimmed = quality.trim_silence(wav, sr)
print(f"trim u15250: {wav.shape[-1]/sr:.2f}s -> {trimmed.shape[-1]/sr:.2f}s shape {tuple(trimmed.shape)}")

importlib.import_module("auk.infer.infer_gradio")
print("import infer_gradio OK")
