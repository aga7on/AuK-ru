"""GigaAM v3 CTC (sherpa-onnx, ONNX int8) transcription wrapper for AuK QC.

Reuses the loader/features from G:\\AI\\kadr\\python\\ttsqc\\gigaam.py (Aligner).
"""
import sys

import numpy as np

KADR_PY = r"G:\AI\kadr\python"
if KADR_PY not in sys.path:
    sys.path.insert(0, KADR_PY)

_ALIGNER = None


def _get_aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from ttsqc.gigaam import Aligner
        _ALIGNER = Aligner()
    return _ALIGNER


def load_16k(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def transcribe(path_or_audio):
    """Wave file path (or 16k mono np array) -> transcribed text."""
    if isinstance(path_or_audio, str):
        audio = load_16k(path_or_audio)
    else:
        audio = np.asarray(path_or_audio, dtype=np.float32)
    al = _get_aligner()
    em = al.emissions(audio)          # [T, 34] log_probs
    ids = em.argmax(axis=1)
    out = []
    prev = None
    for i in ids:
        if i != prev:
            if i != al.blank:
                out.append(al.id2char.get(int(i), ""))
            prev = i
    text = "".join(out).replace("▁", " ").strip()
    return " ".join(text.split())


if __name__ == "__main__":
    import os
    import time

    D = r"G:\AI\AuK\local_tests\autopsy"
    for name in ["ap00_hard_words_s1234.wav", "ap02_hard_words_s7.wav", "ap05_reduction_s1234.wav"]:
        t0 = time.time()
        txt = transcribe(os.path.join(D, name))
        print(f"{name} ({time.time()-t0:.1f}s): {txt}")
