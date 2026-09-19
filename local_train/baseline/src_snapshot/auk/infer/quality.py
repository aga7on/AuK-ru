"""Candidate screening and silence trimming for best-of-N generation.

Screening uses GigaAM v3 CTC (Russian ASR) recall plus a pause-excess metric;
trimming removes over-long silence at edges and inside the waveform.
"""
from __future__ import annotations

import re

import numpy as np
import torch

_ALIGNER = None


def _aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from auk.infer.gigaam_ctc import Aligner

        _ALIGNER = Aligner()
    return _ALIGNER


def _to_mono_np(wav: torch.Tensor) -> np.ndarray:
    x = wav.detach().to(torch.float32).cpu().numpy()
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] <= x.shape[1] else x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def transcribe(wav: torch.Tensor, sr: int) -> str:
    x = _to_mono_np(wav)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return _aligner().transcribe(x)


_WORD_RE = re.compile(r"[^а-яё0-9\s]")


def _words(text: str) -> list[str]:
    text = text.lower().replace("ё", "е").replace("+", "").replace("'", "").replace("-", "")
    text = _WORD_RE.sub(" ", text)
    return [w for w in text.split() if re.search(r"[а-я0-9]", w)]


def recall(expected: str, heard: str) -> float:
    from collections import Counter

    ew, hw = Counter(_words(expected)), Counter(_words(heard))
    if not ew:
        return 1.0
    return sum((ew & hw).values()) / len(ew)


def _speech_mask(x: np.ndarray, sr: int, frame: int = 600, hop: int = 240, rel_db: float = 35.0):
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0:
        return np.zeros(1, dtype=bool), hop / sr
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(r + 1e-9)
    return db > (db.max() - rel_db), hop / sr


def pause_excess(x: np.ndarray, sr: int, lead_max: float = 0.30, tail_max: float = 0.30,
                 int_cap: float = 0.45, min_pause: float = 0.15) -> float:
    speech, hop_s = _speech_mask(x, sr)
    total = len(x) / sr
    lead = 0.0
    for i, s in enumerate(speech):
        if s:
            lead = i * hop_s
            break
    tail = 0.0
    for i in range(len(speech) - 1, -1, -1):
        if speech[i]:
            tail = (len(speech) - 1 - i) * hop_s
            break
    excess = max(0.0, lead - lead_max) + max(0.0, tail - tail_max)
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            d = (i - start) * hop_s
            p = start * hop_s
            if d >= min_pause and p > lead + 0.2 and p < total - tail - 0.2:
                excess += max(0.0, d - int_cap)
            start = None
    return excess


def score_candidate(wav: torch.Tensor, sr: int, expected_text: str) -> tuple[float, float]:
    x = _to_mono_np(wav)
    heard = transcribe(wav, sr)
    return recall(expected_text, heard), pause_excess(x, sr)


def pick_best(candidates: list[tuple[torch.Tensor, int]], expected_text: str) -> int:
    """Return index of the best candidate; falls back to 0 on any failure."""
    try:
        scores = []
        for i, (wav, sr) in enumerate(candidates):
            rec, excess = score_candidate(wav, sr, expected_text)
            scores.append((i, rec, excess))
        best = sorted(scores, key=lambda t: (-t[1], t[2], t[0]))[0]
        return best[0]
    except Exception:
        return 0


def trim_silence(wav: torch.Tensor, sr: int, lead_max: float = 0.15, tail_max: float = 0.20,
                 int_cap: float = 0.45, fade_ms: int = 10) -> torch.Tensor:
    """Shorten over-long lead/tail silence and internal pauses (crossfaded cuts)."""
    x = _to_mono_np(wav)
    speech, hop_s = _speech_mask(x, sr)
    hop = 240
    frame = 600
    keep = np.ones(len(x), dtype=bool)
    runs = []
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            runs.append((start * hop, min(len(x), i * hop + frame), (i - start) * hop_s))
            start = None
    if start is not None:
        runs.append((start * hop, len(x), (len(speech) - start) * hop_s))
    for s0, s1, dur in runs:
        is_lead = s0 == 0
        is_tail = s1 >= len(x) - frame
        if is_lead:
            cut = max(0.0, dur - lead_max)
            if cut > 0:
                keep[s0:s0 + int(cut * sr)] = False
        elif is_tail:
            cut = max(0.0, dur - tail_max)
            if cut > 0:
                keep[s1 - int(cut * sr):s1] = False
        else:
            cut = max(0.0, dur - int_cap)
            if cut > 0:
                mid = (s0 + s1) // 2
                half = int(cut * sr) // 2
                a, b = max(0, mid - half), min(len(x), mid + half)
                keep[a:b] = False
    if not keep.all():
        fade = max(1, int(sr * fade_ms / 1000))
        edges = np.flatnonzero(keep[1:] != keep[:-1]) + 1
        for e in edges:
            if keep[e]:
                x[e:e + fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            else:
                x[max(0, e - fade):e] *= np.linspace(1.0, 0.0, e - max(0, e - fade), dtype=np.float32)
    out = x[keep]
    if out.size == 0:
        return wav
    out_t = torch.from_numpy(out)
    if wav.ndim == 2:
        out_t = out_t.reshape(1, -1) if wav.shape[0] == 1 else out_t.reshape(-1, 1)
    return out_t
