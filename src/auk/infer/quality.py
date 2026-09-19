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
    text = text.lower().replace("ё", "е").replace("+", "").replace("'", "")
    text = text.replace("-", " ")
    text = _WORD_RE.sub(" ", text)
    return [w for w in text.split() if re.search(r"[а-я0-9]", w)]


def recall(expected: str, heard: str) -> float:
    from collections import Counter

    ew, hw = Counter(_words(expected)), Counter(_words(heard))
    total = sum(ew.values())
    if total == 0:
        return 1.0
    matched = sum((ew & hw).values())
    return min(1.0, matched / total)


def _align_words(ref: list[str], hyp: list[str]) -> list[str]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append("=" if ref[i - 1] == hyp[j - 1] else "S")
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append("D")
            i -= 1
        else:
            ops.append("I")
            j -= 1
    return ops[::-1]


def wer_metrics(expected: str, heard: str) -> dict:
    """WER по словам (S/D/I), повторы и длины — фильтр содержания."""
    rw, hw = _words(expected), _words(heard)
    ops = _align_words(rw, hw)
    subs = ops.count("S")
    dels = ops.count("D")
    ins = ops.count("I")
    hits = ops.count("=")
    dupes = sum(1 for i in range(1, len(hw)) if hw[i] == hw[i - 1])
    return {"wer": (subs + dels + ins) / max(len(rw), 1), "subs": subs, "dels": dels,
            "ins": ins, "hits": hits, "dupes": dupes, "ref_n": len(rw), "hyp_n": len(hw)}


def clip_ratio(x) -> float:
    x = np.asarray(x)
    return float(np.mean(np.abs(x) >= 0.985)) if x.size else 0.0


def score_candidate(wav: torch.Tensor, sr: int, expected_text: str) -> tuple[float, float, float]:
    """(WER, clip_ratio, pause_excess) — метрика отбора кандидата."""
    x = _to_mono_np(wav)
    heard = transcribe(wav, sr)
    m = wer_metrics(expected_text, heard)
    return m["wer"], clip_ratio(x), pause_excess(x, sr)


def pick_best(candidates: list[tuple[torch.Tensor, int]], expected_text: str) -> int:
    """Индекс лучшего кандидата: WER -> клиппинг -> паузы; fallback 0 (явно)."""
    try:
        scored = []
        for i, (wav, sr) in enumerate(candidates):
            wer, clip, excess = score_candidate(wav, sr, expected_text)
            scored.append((i, wer, clip > 5e-4, clip, excess))
        best = sorted(scored, key=lambda t: (t[1], t[2], t[3], t[4], t[0]))[0]
        return best[0]
    except Exception:
        return 0


def limit_peak(wav: torch.Tensor, ceiling: float = 0.95) -> torch.Tensor:
    """Мягкий потолок амплитуды: масштабирует, если пик выше ceiling (анти-клиппинг)."""
    x = wav.detach().to(torch.float32)
    peak = float(x.abs().max()) if x.numel() else 0.0
    if peak > ceiling > 0:
        x = x * (ceiling / peak)
    return x


def normalize_rms(wav: torch.Tensor, target_dbfs: float = -20.0,
                  floor_dbfs: float = -40.0, ceiling: float = 0.95) -> torch.Tensor:
    """Нормализация громкости по RMS активной речи.

    Приводит RMS к target_dbfs (по кадрам активной речи, чтобы тишина по краям не
    занижала оценку), затем применяет limit_peak. Ничего не делает, если сигнал
    тише floor_dbfs (шум/пустышку не усиливаем). Модель не управляет громкостью —
    RMS определяется сидом (замер 18.09: спред до 11 dB между сидами), поэтому
    нормализация — обязательный пост-процесс для консистентного прослушивания.
    """
    x = wav.detach().to(torch.float32)
    if x.numel() == 0:
        return x
    np_x = _to_mono_np(x)
    sr = 24000  # только для маски речи; шаг кадра не влияет на выбор активных
    speech, _ = _speech_mask(np_x, sr)
    idx = np.flatnonzero(speech)
    if idx.size:
        hop = 240
        frame = 600
        sel = np.concatenate([np.arange(i * hop, min(len(np_x), i * hop + frame)) for i in idx[:200]])
        sel = np.unique(sel)
        if sel.size:
            np_x_act = np_x[sel]
        else:
            np_x_act = np_x
    else:
        np_x_act = np_x
    rms = float(np.sqrt((np_x_act ** 2).mean()) + 1e-12)
    target = 10 ** (target_dbfs / 20)
    if rms < 10 ** (floor_dbfs / 20):
        return limit_peak(x, ceiling)
    scale = target / rms
    if wav.ndim == 2 and wav.shape[0] == 1:
        x = x * scale
    else:
        x = x * scale
    return limit_peak(x, ceiling)


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
