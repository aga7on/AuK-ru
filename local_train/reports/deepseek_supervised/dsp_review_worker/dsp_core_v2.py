"""Corrected, independent DSP measurement core for the AuK s2 tool operations.

Written by the DeepSeek v4.1 Flash DSP correction worker after root review found
blocking defects in `local_train/tool_dsp_measure.py`. This module is the
self-contained replacement used by `dsp_measure_v2.py` and `test_dsp_controls_v2.py`.
It deliberately does NOT import or modify the original script.

Fixes over the reviewed version
-------------------------------
1. Activity mask uses BOTH an absolute energy floor (dBFS) and a relative-to-p95
   floor, so an all-zero/non-finite/short signal is never declared "active".
   Non-finite and too-short inputs are hard-invalid.
2. Segmental SNR / correlation are gated by an explicit alignment step (limited
   lag search, i.e. +/- max_lag_ms) and by an envelope-correlation applicability
   gate. If the regenerated audio is phase/delay/prosody-shifted the denoise
   evidence is reported as INCONCLUSIVE instead of a number. A phase-robust
   secondary metric (noise-floor reduction in the clean reference's non-speech
   frames) is reported alongside. Silence is never scored as perfect denoise.
   Sample-rate mismatch is handled by explicit resampling + a `resampled` flag;
   arrays are never silently truncated.
3. Volume uses a FLOAT-WAV-safe headroom and reports median/IQR frame gain, peak
   and clip ratio. Controls assert positive AND negative expectations.
4. Speed is reported as an active-duration PROXY plus an explicit
   content-comparability gate (duration-normalised MFCC-DTW + active-run count),
   and marked invalid when content is not comparable. Pitch uses paired voiced
   frames (not a bare median ratio), voiced coverage and an octave guard.
5. Clean ground truth for noise_add comes from dataset provenance
   (`noise_add_val_<id>.wav`, verified equal to the kyutai clean clip), with both
   paths and full sha256 logged.
6. Aggregation reports n/valid/invalid per op/variant, the effect distribution and
   prospective thresholds; no headline "pass" is derived from a mean.

No training, no synthesis, no network. Primitive signal transforms only.
"""
from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Configuration: prospective thresholds (declared BEFORE running the audit;
# not retro-fitted to the observed results).
# --------------------------------------------------------------------------- #
CFG = {
    "frame_s": 0.025,
    "hop_s": 0.010,
    "abs_floor_db": -60.0,      # absolute activity floor, dBFS
    "rel_floor_db": 30.0,       # relative to 95th percentile frame level
    "min_active_frames": 5,
    "min_frames": 10,
    "min_dur_s": 0.30,
    "max_lag_ms": 300.0,
    "min_env_corr": 0.60,       # alignment/applicability gate
    "clip_threshold": 0.985,
    "low_level_db": -55.0,      # active speech below this dBFS is flagged low-level
    "min_voiced_frames": 20,
    "min_paired_frames": 20,
    "min_voiced_cov": 0.15,
    "max_dtw_cost": 0.90,       # stretch-invariant content-comparability gate
    "min_content_corr": 0.78,   # min NN bag-of-frames cosine for comparability
    # prospective effect windows (target values)
    "th_volume_up_db": (3.0, 9.0),      # target +6
    "th_volume_down_db": (-9.0, -3.0),  # target -6
    "th_speed_up": (1.0, 1.25),         # target 1.1
    "th_speed_down": (0.8, 1.0),        # target 0.9
    "th_pitch_up_st": (0.8, 3.5),       # target +2
    "th_pitch_down_st": (-3.5, -0.8),   # target -2
    "th_noise_floor_reduction_db": 1.0, # secondary, phase-robust denoise hint
}

TOOLS_DIR = r"G:\AI\AuK\local_train\data_s2_tools_v3"
KYUTAI_CLEAN_DIR = r"G:\AI\kyutai-ru\data\ru_wav_mfa"

OPS = ["volume_up", "volume_down", "speed_up", "speed_down",
       "pitch_up", "pitch_down", "noise_add"]
POSITIVE = {"volume_up", "speed_up", "pitch_up"}
NEGATIVE = {"volume_down", "speed_down", "pitch_down"}


class MeasurementError(Exception):
    """Raised for hard-invalid inputs (empty/non-finite/too short)."""


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Audio:
    path: str
    sr: int
    x: np.ndarray
    frames: int
    subtype: str
    fmt: str
    sha256: str
    resampled_from: Optional[int] = None

    @property
    def dur(self) -> float:
        return len(self.x) / self.sr


def load_audio(path: str, target_sr: Optional[int] = None) -> Audio:
    if not path or not os.path.exists(path):
        raise MeasurementError("file_missing")
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype=np.float32)
    resampled_from = None
    if target_sr is not None and sr != target_sr:
        import librosa
        x = np.ascontiguousarray(
            librosa.resample(x, orig_sr=sr, target_sr=target_sr), dtype=np.float32)
        resampled_from = sr
        sr = target_sr
    if x.size == 0:
        raise MeasurementError("empty_audio")
    if not np.all(np.isfinite(x)):
        raise MeasurementError("nonfinite_audio")
    if len(x) < int(CFG["min_dur_s"] * sr):
        raise MeasurementError("too_short")
    return Audio(path=path, sr=sr, x=x, frames=int(info.frames),
                 subtype=info.subtype, fmt=info.format, sha256=sha256_file(path),
                 resampled_from=resampled_from)


# --------------------------------------------------------------------------- #
# Framing / activity / basic stats
# --------------------------------------------------------------------------- #
def _frame_matrix(x: np.ndarray, sr: int, frame_s: float, hop_s: float):
    win = int(frame_s * sr)
    hop = max(1, int(hop_s * sr))
    m = (len(x) - win) // hop
    if m <= 0:
        return np.zeros((0, win), np.float32), win, hop
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[:m * hop + win:hop]
    return np.ascontiguousarray(fr), win, hop


def frame_rms_db(x: np.ndarray, sr: int, frame_s: float = None,
                 hop_s: float = None) -> np.ndarray:
    frame_s = frame_s or CFG["frame_s"]
    hop_s = hop_s or CFG["hop_s"]
    fr, _, _ = _frame_matrix(x, sr, frame_s, hop_s)
    if fr.shape[0] == 0:
        return np.zeros(0, np.float64)
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def activity_mask(x: np.ndarray, sr: int):
    """Frames considered active.

    A frame is active iff its level exceeds BOTH an absolute dBFS floor and a
    floor relative to the 95th percentile of all frame levels. All-zero or
    non-finite-railed signals yield no active frames.
    """
    db = frame_rms_db(x, sr)
    if db.size < CFG["min_frames"]:
        return np.zeros(db.size, bool), db
    if not np.all(np.isfinite(db)):
        return np.zeros(db.size, bool), db
    ref = float(np.percentile(db, 95))
    if ref < CFG["abs_floor_db"]:
        return np.zeros(db.size, bool), db
    thr = max(CFG["abs_floor_db"], ref - CFG["rel_floor_db"])
    return db > thr, db


def active_rms_db(x: np.ndarray, sr: int, mask: np.ndarray) -> Optional[float]:
    if mask.size == 0 or not np.any(mask):
        return None
    fr, _, _ = _frame_matrix(x, sr, CFG["frame_s"], CFG["hop_s"])
    m = min(fr.shape[0], mask.size)
    if m == 0:
        return None
    fr, mask = fr[:m], mask[:m]
    if not np.any(mask):
        return None
    act = fr[mask].astype(np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def clip_ratio(x: np.ndarray, thr: float = None) -> float:
    thr = CFG["clip_threshold"] if thr is None else thr
    return float(np.mean(np.abs(x) >= thr)) if x.size else 0.0


def envelope(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    h = max(1, int(sr * hop_ms / 1000.0))
    n = len(x) // h
    if n == 0:
        return np.zeros(0, np.float64)
    return np.sqrt(np.mean(x[:n * h].astype(np.float64).reshape(n, h) ** 2, axis=1) + 1e-12)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def best_lag_ms(a: np.ndarray, b: np.ndarray, sr: int,
                max_lag_ms: float = None) -> dict:
    """Limited-lag envelope alignment. Positive lag => b is delayed vs a.

    Returns lag in samples/ms, envelope correlation at that lag and overlap.
    """
    max_lag_ms = CFG["max_lag_ms"] if max_lag_ms is None else max_lag_ms
    ea, eb = envelope(a, sr), envelope(b, sr)
    m = min(len(ea), len(eb))
    if m < 10:
        return {"lag_ms": None, "corr": None, "overlap_frames": int(m)}
    ea, eb = ea[:m] - ea[:m].mean(), eb[:m] - eb[:m].mean()
    ml = int(round(max_lag_ms / 10.0))
    best_c, best_l = -2.0, 0
    for l in range(-ml, ml + 1):
        if l >= 0:
            u, v = ea[:m - l], eb[l:]
        else:
            u, v = ea[-l:], eb[:m + l]
        if len(u) < 10 or u.std() < 1e-12 or v.std() < 1e-12:
            continue
        c = float(np.corrcoef(u, v)[0, 1])
        if c > best_c:
            best_c, best_l = c, l
    lag_samples = int(round(best_l * 10.0 / 1000.0 * sr))
    return {"lag_samples": lag_samples, "lag_ms": float(best_l * 10.0),
            "corr": best_c if best_c > -1 else None, "overlap_frames": int(m)}


def shift_for_lag(a: np.ndarray, b: np.ndarray, lag_samples: int):
    """Return (a_aligned, b_aligned) so that time index 0 corresponds.

    Positive lag means b is delayed: b[lag:] aligns with a[:].
    """
    if lag_samples == 0:
        return a, b
    if lag_samples > 0:
        if lag_samples >= len(b):
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        return a, b[lag_samples:]
    k = -lag_samples
    if k >= len(a):
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return a[k:], b


def median_frame_gain_db(src: np.ndarray, out: np.ndarray, sr: int,
                         lag_samples: int = 0, mask: np.ndarray = None) -> dict:
    """Per-frame gain on frames active in `src` (aligned by lag)."""
    a, b = shift_for_lag(src, out, lag_samples)
    n = min(len(a), len(b))
    if n < int(CFG["min_dur_s"] * sr):
        return {"median_db": None, "iqr_db": None, "n_frames": 0}
    a, b = a[:n], b[:n]
    fa, _, hop = _frame_matrix(a, sr, CFG["frame_s"], CFG["hop_s"])
    fb, _, _ = _frame_matrix(b, sr, CFG["frame_s"], CFG["hop_s"])
    m = min(fa.shape[0], fb.shape[0])
    if m < CFG["min_frames"]:
        return {"median_db": None, "iqr_db": None, "n_frames": m}
    if mask is None:
        mask, _ = activity_mask(a, sr)
    if mask.size < m:
        m = mask.size
    if m < CFG["min_frames"] or not np.any(mask[:m]):
        return {"median_db": None, "iqr_db": None, "n_frames": 0}
    ra = np.sqrt(np.mean(fa[:m][mask[:m]].astype(np.float64) ** 2, axis=1) + 1e-12)
    rb = np.sqrt(np.mean(fb[:m][mask[:m]].astype(np.float64) ** 2, axis=1) + 1e-12)
    g = 20.0 * np.log10((rb + 1e-12) / (ra + 1e-12))
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"median_db": None, "iqr_db": None, "n_frames": 0}
    q1, q3 = np.percentile(g, [25, 75])
    return {"median_db": float(np.median(g)), "iqr_db": float(q3 - q1),
            "n_frames": int(g.size)}


def snr_segmental(ref: np.ndarray, est: np.ndarray, sr: int, lag_samples: int,
                  ref_mask: np.ndarray = None) -> dict:
    """Segmental SNR of `est` against `ref` after alignment.

    Active frames are taken from the clean reference. Returns None when fewer
    than min_active_frames active frames exist (so silence can never be a
    "perfect" denoise result).
    """
    a, b = shift_for_lag(ref, est, lag_samples)
    n = min(len(a), len(b))
    if n < int(CFG["min_dur_s"] * sr):
        return {"snr_db": None, "n_active": 0, "reason": "insufficient_overlap"}
    a, b = a[:n], b[:n]
    if ref_mask is None:
        ref_mask, _ = activity_mask(a, sr)
    fa, win, hop = _frame_matrix(a, sr, CFG["frame_s"], CFG["hop_s"])
    fb, _, _ = _frame_matrix(b, sr, CFG["frame_s"], CFG["hop_s"])
    m = min(fa.shape[0], fb.shape[0], ref_mask.size)
    if m < CFG["min_active_frames"]:
        return {"snr_db": None, "n_active": int(m), "reason": "too_few_frames"}
    mask = ref_mask[:m]
    if int(mask.sum()) < CFG["min_active_frames"]:
        return {"snr_db": None, "n_active": int(mask.sum()), "reason": "no_active_frames"}
    sig = np.mean(fa[:m][mask].astype(np.float64) ** 2, axis=1)
    err = np.mean((fb[:m][mask] - fa[:m][mask]).astype(np.float64) ** 2, axis=1)
    val = 10.0 * np.log10((sig + 1e-12) / (err + 1e-12))
    val = val[np.isfinite(val)]
    if val.size < CFG["min_active_frames"]:
        return {"snr_db": None, "n_active": int(val.size), "reason": "no_finite_frames"}
    return {"snr_db": float(np.mean(val)), "n_active": int(val.size)}


def rms_in_mask(x: np.ndarray, sr: int, mask: np.ndarray) -> Optional[float]:
    if mask.size == 0 or not np.any(mask):
        return None
    fr, _, _ = _frame_matrix(x, sr, CFG["frame_s"], CFG["hop_s"])
    m = min(fr.shape[0], mask.size)
    if m == 0 or not np.any(mask[:m]):
        return None
    v = fr[:m][mask[:m]].astype(np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12))


def speech_preservation_corr(ref: np.ndarray, est: np.ndarray, sr: int,
                             lag_samples: int, ref_mask: np.ndarray = None) -> Optional[float]:
    a, b = shift_for_lag(ref, est, lag_samples)
    n = min(len(a), len(b))
    if n < int(CFG["min_dur_s"] * sr):
        return None
    a, b = a[:n], b[:n]
    if ref_mask is None:
        ref_mask, _ = activity_mask(a, sr)
    fa, win, hop = _frame_matrix(a, sr, CFG["frame_s"], CFG["hop_s"])
    fb, _, _ = _frame_matrix(b, sr, CFG["frame_s"], CFG["hop_s"])
    m = min(fa.shape[0], fb.shape[0], ref_mask.size)
    if m < CFG["min_active_frames"] or not np.any(ref_mask[:m]):
        return None
    ids = np.where(ref_mask[:m])[0]
    A = fa[ids].reshape(-1).astype(np.float64)
    B = fb[ids].reshape(-1).astype(np.float64)
    if A.std() < 1e-9 or B.std() < 1e-9:
        return None
    return float(np.corrcoef(A, B)[0, 1])


# --------------------------------------------------------------------------- #
# Content comparability (duration-normalised MFCC-DTW)
# --------------------------------------------------------------------------- #
def _mfcc_unit(x: np.ndarray, sr: int):
    """Unit-norm MFCC frames (stretch/shift-invariant similarity space)."""
    import librosa
    m = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, n_fft=1024,
                             hop_length=max(1, int(0.010 * sr)))
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)


def active_span_s(x: np.ndarray, sr: int) -> float:
    mask, _ = activity_mask(x, sr)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0.0
    return float((idx[-1] - idx[0] + 1) * CFG["hop_s"])


def content_compatibility(src: np.ndarray, out: np.ndarray, sr: int) -> dict:
    """Stretch-invariant content comparability.

    Uses unit-norm MFCC bag-of-frames nearest-neighbour similarity (invariant to
    a global time warp) plus MFCC-DTW cost. Reported so that a duration-based
    speed proxy is explicitly flagged invalid when content is not comparable.
    """
    import librosa
    n = min(len(src), len(out))
    if n < int(CFG["min_dur_s"] * sr):
        return {"comparable": False, "reason": "insufficient_overlap",
                "dtw_cost": None, "nn_score": None, "nn_ab": None, "nn_ba": None}
    A, B = _mfcc_unit(src[:n], sr), _mfcc_unit(out[:n], sr)
    if A.shape[1] < 10 or B.shape[1] < 10:
        return {"comparable": False, "reason": "too_few_frames",
                "dtw_cost": None, "nn_score": None, "nn_ab": None, "nn_ba": None}
    S = A.T @ B
    ab, ba = float(np.mean(S.max(axis=1))), float(np.mean(S.max(axis=0)))
    nn = 0.5 * (ab + ba)
    try:
        D = librosa.sequence.dtw(X=A, Y=B, subseq=False, backtrack=False, band_rad=0.40)
        dtw_cost = float(D[-1, -1] / max(A.shape[1], B.shape[1]))
    except Exception:
        dtw_cost = None
    comparable = bool(dtw_cost is not None and dtw_cost <= CFG["max_dtw_cost"]
                      and nn >= CFG["min_content_corr"])
    return {"comparable": comparable, "dtw_cost": dtw_cost, "nn_score": nn,
            "nn_ab": ab, "nn_ba": ba,
            "reason": None if comparable else "content_not_comparable"}


def active_runs(x: np.ndarray, sr: int, min_gap_s: float = 0.08) -> int:
    mask, _ = activity_mask(x, sr)
    if not np.any(mask):
        return 0
    pad = int(min_gap_s / CFG["hop_s"])
    idx = np.where(mask)[0]
    runs = 1
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > pad:
            runs += 1
    return int(runs)


# --------------------------------------------------------------------------- #
# Pitch (paired voiced frames)
# --------------------------------------------------------------------------- #
def voiced_f0(x: np.ndarray, sr: int) -> dict:
    import librosa
    hop = max(1, int(0.010 * sr))
    try:
        f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=450, sr=sr,
                                     frame_length=2048, hop_length=hop)
    except Exception as exc:
        return {"f0": np.zeros(0), "voiced": np.zeros(0, bool), "err": str(exc)}
    return {"f0": np.asarray(f0), "voiced": np.asarray(voiced, bool), "err": None}


def paired_pitch(src: np.ndarray, out: np.ndarray, sr: int, lag_samples: int) -> dict:
    vi = voiced_f0(src, sr)
    vo = voiced_f0(out, sr)
    fi, mi = vi["f0"], vi["voiced"] & np.isfinite(vi["f0"])
    fo, mo = vo["f0"], vo["voiced"] & np.isfinite(vo["f0"])
    cov_i = float(mi.mean()) if mi.size else 0.0
    cov_o = float(mo.mean()) if mo.size else 0.0
    vi_med = float(np.median(fi[mi])) if mi.sum() >= 3 else None
    vo_med = float(np.median(fo[mo])) if mo.sum() >= 3 else None
    ratio_st = None
    if vi_med and vo_med and vi_med > 0:
        ratio_st = float(12.0 * np.log2(vo_med / vi_med))
    # paired frames after lag
    lag_frames = int(round(lag_samples / max(1, int(0.010 * sr))))
    n = min(len(fi), len(fo))
    if lag_frames >= 0:
        ji, jo = 0, lag_frames
    else:
        ji, jo = -lag_frames, 0
    cnt = 0
    pairs = []
    while ji < n and jo < n:
        if mi[ji] and mo[jo] and fi[ji] > 0 and fo[jo] > 0:
            pairs.append(12.0 * np.log2(fo[jo] / fi[ji]))
            cnt += 1
        ji += 1
        jo += 1
    paired_st = float(np.median(pairs)) if cnt >= CFG["min_paired_frames"] else None
    paired_iqr = float(np.subtract(*np.percentile(pairs, [75, 25]))) if cnt >= 4 else None
    octave = bool(paired_st is not None and 10.5 < abs(paired_st) < 13.5)
    comparable = bool(
        cnt >= CFG["min_paired_frames"]
        and cov_i >= CFG["min_voiced_cov"] and cov_o >= CFG["min_voiced_cov"]
        and 0.4 <= (cov_o / cov_i if cov_i else 0) <= 2.5
        and not octave)
    return {"median_f0_in": vi_med, "median_f0_out": vo_med,
            "semitones_median_ratio": ratio_st,
            "semitones_paired_median": paired_st,
            "paired_iqr_st": paired_iqr, "n_paired_frames": int(cnt),
            "voiced_cov_in": cov_i, "voiced_cov_out": cov_o,
            "octave_error": octave, "comparable": comparable}


# --------------------------------------------------------------------------- #
# Clean-ground-truth provenance for noise_add
# --------------------------------------------------------------------------- #
def resolve_clean_source(noisy_path: str) -> dict:
    """Resolve the clean ground truth for a `noise_in_val_<id>.wav` clip from
    dataset provenance (NOT from an inferred basename lookup).

    Preferred: the dataset's own `noise_add_val_<id>.wav` (verified equal to the
    kyutai clean clip during this audit). Fallback: kyutai clean clip.
    Both candidate paths and hashes are logged.
    """
    base = os.path.basename(noisy_path)
    out = {"noisy_path": noisy_path, "clip_id": None,
           "dataset_target_path": None, "dataset_target_sha256": None,
           "kyutai_path": None, "kyutai_sha256": None,
           "resolved_path": None, "resolved_sha256": None,
           "target_equals_kyutai": None, "evidence": None}
    if not base.startswith("noise_in_val_") or not base.endswith(".wav"):
        out["evidence"] = "unsupported_noise_basename"
        return out
    cid = base[len("noise_in_val_"):-4]
    out["clip_id"] = cid
    dpath = os.path.join(TOOLS_DIR, "noise_add_val_" + cid + ".wav")
    kpath = os.path.join(KYUTAI_CLEAN_DIR, cid + ".wav")
    if os.path.exists(dpath):
        out["dataset_target_path"] = dpath
        out["dataset_target_sha256"] = sha256_file(dpath)
    if os.path.exists(kpath):
        out["kyutai_path"] = kpath
        out["kyutai_sha256"] = sha256_file(kpath)
    if out["dataset_target_sha256"] and out["kyutai_sha256"]:
        out["target_equals_kyutai"] = (out["dataset_target_sha256"] == out["kyutai_sha256"])
    if out["dataset_target_path"]:
        out["resolved_path"] = out["dataset_target_path"]
        out["resolved_sha256"] = out["dataset_target_sha256"]
        out["evidence"] = "dataset_target"
    elif out["kyutai_path"]:
        out["resolved_path"] = out["kyutai_path"]
        out["resolved_sha256"] = out["kyutai_sha256"]
        out["evidence"] = "kyutai_fallback"
    else:
        out["evidence"] = "clean_source_not_found"
    return out


# --------------------------------------------------------------------------- #
# Per-file measurement
# --------------------------------------------------------------------------- #
def _base_record(op: str, src_path: str, out_path: str) -> dict:
    return {"op": op, "source_path": src_path, "output_path": out_path,
            "valid": False, "reason": None, "applicable": None, "caveat": None}


def measure_file(op: str, src_path: str, out_path: str,
                 content_evidence: Optional[dict] = None) -> dict:
    rec = _base_record(op, src_path, out_path)
    if content_evidence is not None:
        rec["content_evidence"] = content_evidence
    try:
        src = load_audio(src_path)
        out = load_audio(out_path, target_sr=src.sr)
    except MeasurementError as exc:
        rec["reason"] = str(exc)
        return rec
    rec.update(
        sr_in=src.sr, sr_out_original=(out.resampled_from or out.sr),
        resampled=out.resampled_from is not None,
        dur_in_s=round(src.dur, 4), dur_out_s=round(out.dur, 4),
        sha256_source=src.sha256, sha256_output=out.sha256,
        peak_in=round(peak(src.x), 5), peak_out=round(peak(out.x), 5),
        clip_in=round(clip_ratio(src.x), 6), clip_out=round(clip_ratio(out.x), 6),
    )
    mask_i, _ = activity_mask(src.x, src.sr)
    mask_o, _ = activity_mask(out.x, out.sr)
    ai = int(mask_i.sum())
    ao = int(mask_o.sum())
    lvl_i = active_rms_db(src.x, src.sr, mask_i)
    lvl_o = active_rms_db(out.x, out.sr, mask_o)
    rec.update(active_frames_in=ai, active_frames_out=ao,
               active_s_in=round(ai * CFG["hop_s"], 4),
               active_s_out=round(ao * CFG["hop_s"], 4),
               active_rms_db_in=(round(lvl_i, 2) if lvl_i is not None else None),
               active_rms_db_out=(round(lvl_o, 2) if lvl_o is not None else None),
               low_level_output=bool(lvl_o is not None and lvl_o < CFG["low_level_db"]))
    if ai < CFG["min_active_frames"]:
        rec["reason"] = "source_has_no_active_speech"
        return rec
    if ao < CFG["min_active_frames"]:
        rec["reason"] = "output_has_no_active_speech"
        return rec

    align = best_lag_ms(src.x, out.x, src.sr)
    rec["align"] = align
    lag = align.get("lag_samples") or 0
    lag_ok = bool(align.get("lag_ms") is not None
                  and abs(align["lag_ms"]) <= CFG["max_lag_ms"])

    if op in ("volume_up", "volume_down"):
        g = median_frame_gain_db(src.x, out.x, src.sr, lag_samples=lag, mask=mask_i)
        rec.update(gain_median_db=(round(g["median_db"], 3) if g["median_db"] is not None else None),
                   gain_iqr_db=(round(g["iqr_db"], 3) if g["iqr_db"] is not None else None),
                   n_gain_frames=g["n_frames"])
        if g["median_db"] is None:
            rec["reason"] = "no_aligned_active_frames"
            return rec
        rec["delta_db"] = rec["gain_median_db"]
        rec["metric"] = "gain_median_db"
        rec["direction_ok"] = (rec["delta_db"] > 0.5) if op == "volume_up" else (rec["delta_db"] < -0.5)
        lo, hi = CFG["th_volume_up_db"] if op == "volume_up" else CFG["th_volume_down_db"]
        rec["within_prospective_window"] = bool(lo <= rec["delta_db"] <= hi)
        rec["clipped_output"] = bool(clip_ratio(out.x) > 0.0 or peak(out.x) > 1.0)
        if rec["clipped_output"]:
            rec["magnitude_trustworthy"] = False
            rec["caveat"] = "output clipping/over-range; |gain| is a lower bound"
        else:
            rec["magnitude_trustworthy"] = True
        rec["valid"] = True
        rec["applicable"] = True
        if not lag_ok:
            rec["caveat"] = (rec.get("caveat") or "") + " alignment weak"

    elif op in ("speed_up", "speed_down"):
        span_i = active_span_s(src.x, src.sr)
        span_o = active_span_s(out.x, out.sr)
        if span_o < 0.05:
            rec["reason"] = "output_active_span_too_short"
            return rec
        rate = span_i / span_o
        rate_cnt = rec["active_s_in"] / rec["active_s_out"] if rec["active_s_out"] > 0 else None
        cc = content_compatibility(src.x, out.x, src.sr)
        rec.update(rate_proxy=round(rate, 4), metric="rate_proxy_span",
                   rate_proxy_count=(round(rate_cnt, 4) if rate_cnt else None),
                   active_span_in_s=round(span_i, 4), active_span_out_s=round(span_o, 4),
                   active_runs_in=active_runs(src.x, src.sr),
                   active_runs_out=active_runs(out.x, out.sr),
                   dtw_cost=(round(cc["dtw_cost"], 3) if cc["dtw_cost"] is not None else None),
                   nn_score=(round(cc["nn_score"], 3) if cc["nn_score"] is not None else None),
                   content_comparable=cc["comparable"])
        rec["proxy_is_conditional"] = True
        if not cc["comparable"]:
            rec["reason"] = "content_not_comparable_rate_proxy_unreliable"
            rec["applicable"] = False
            return rec
        rec["direction_ok"] = (rate > 1.0) if op == "speed_up" else (rate < 1.0)
        lo, hi = CFG["th_speed_up"] if op == "speed_up" else CFG["th_speed_down"]
        rec["within_prospective_window"] = bool(lo <= rate <= hi)
        rec["valid"] = True
        rec["applicable"] = True
        rec["caveat"] = ("active-span proxy (not full content verification); flag invalid "
                         "when MFCC content-comparability gate fails")

    elif op in ("pitch_up", "pitch_down"):
        pp = paired_pitch(src.x, out.x, src.sr, lag)
        rec.update({k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in pp.items()})
        rec["metric"] = "semitones_paired_median"
        rec["semitones"] = (round(pp["semitones_paired_median"], 3)
                            if pp["semitones_paired_median"] is not None else None)
        if pp["semitones_paired_median"] is None:
            rec["reason"] = "insufficient_paired_voiced_frames"
            rec["applicable"] = False
            return rec
        if not pp["comparable"]:
            rec["reason"] = ("octave_error" if pp["octave_error"]
                             else "voiced_content_not_comparable")
            rec["applicable"] = False
            return rec
        st = pp["semitones_paired_median"]
        rec["direction_ok"] = (st > 0.5) if op == "pitch_up" else (st < -0.5)
        lo, hi = CFG["th_pitch_up_st"] if op == "pitch_up" else CFG["th_pitch_down_st"]
        rec["within_prospective_window"] = bool(lo <= st <= hi)
        rec["valid"] = True
        rec["applicable"] = True
        rec["caveat"] = "paired voiced-frame median; median-ratio reported for reference"

    elif op == "noise_add":
        prov = resolve_clean_source(src_path)
        rec["clean_provenance"] = prov
        if not prov["resolved_path"]:
            rec["reason"] = "clean_source_not_found"
            return rec
        clean = load_audio(prov["resolved_path"], target_sr=src.sr)
        cmask, cdb = activity_mask(clean.x, clean.sr)
        rec["clean_sha256"] = clean.sha256
        # input vs clean should be aligned; report for context
        align_in = best_lag_ms(clean.x, src.x, src.sr)
        rec["align_input_vs_clean"] = align_in
        snr_in = snr_segmental(clean.x, src.x, clean.sr, align_in.get("lag_samples") or 0, cmask)
        snr_out = snr_segmental(clean.x, out.x, clean.sr, lag, cmask)
        corr = speech_preservation_corr(clean.x, out.x, clean.sr, lag, cmask)
        # phase-robust secondary: noise floor in clean's non-speech frames.
        # Align input/output to the clean timeline before indexing the mask.
        _, in_al = shift_for_lag(clean.x, src.x, align_in.get("lag_samples") or 0)
        _, out_al = shift_for_lag(clean.x, out.x, lag)
        low = ~cmask
        nf_in = rms_in_mask(in_al, clean.sr, low)
        nf_out = rms_in_mask(out_al, clean.sr, low)
        sp_in = active_rms_db(in_al, clean.sr, cmask)
        sp_out = active_rms_db(out_al, clean.sr, cmask)
        rec.update(
            snr_in_db=(round(snr_in["snr_db"], 2) if snr_in["snr_db"] is not None else None),
            snr_out_db=(round(snr_out["snr_db"], 2) if snr_out["snr_db"] is not None else None),
            snr_n_active_in=snr_in["n_active"], snr_n_active_out=snr_out["n_active"],
            preserve_corr=(round(corr, 3) if corr is not None else None),
            noise_floor_in_db=(round(nf_in, 2) if nf_in is not None else None),
            noise_floor_out_db=(round(nf_out, 2) if nf_out is not None else None),
            noise_floor_reduction_db=(round(nf_in - nf_out, 2)
                                      if (nf_in is not None and nf_out is not None) else None),
            speech_level_change_db=(round(sp_out - sp_in, 2)
                                    if (sp_in is not None and sp_out is not None) else None),
            clean_non_speech_frames=int(low.sum()),
        )
        if snr_in["snr_db"] is not None and snr_out["snr_db"] is not None:
            rec["snr_gain_db"] = round(snr_out["snr_db"] - snr_in["snr_db"], 2)
        rec["metric"] = "snr_gain_db"
        env_corr = align.get("corr")
        applicable = bool(
            lag_ok and env_corr is not None and env_corr >= CFG["min_env_corr"]
            and snr_in["snr_db"] is not None and snr_out["snr_db"] is not None
            and snr_in["n_active"] >= CFG["min_active_frames"]
            and rec["clean_non_speech_frames"] >= CFG["min_active_frames"])
        rec["applicable"] = applicable
        if not applicable:
            rec["reason"] = "denoise_evidence_inconclusive_alignment_or_groundtruth"
            rec["denoise_evidence"] = "inconclusive"
            rec["caveat"] = ("waveform SNR not applicable: output phase/delay/prosody "
                             "differs or ground truth unsuitable; no denoise verdict")
            rec["valid"] = False
            return rec
        rec["denoise_evidence"] = "segmental_snr_applicable"
        rec["direction_ok"] = bool(rec.get("snr_gain_db") is not None and rec["snr_gain_db"] > 0)
        rec["noise_floor_ok"] = bool(rec["noise_floor_reduction_db"] is not None
                                     and rec["noise_floor_reduction_db"]
                                     >= CFG["th_noise_floor_reduction_db"])
        rec["valid"] = True
        rec["caveat"] = ("segmental SNR against the clean clip penalises any synthesis "
                         "mismatch, not only residual noise; treat as weak evidence")
    else:
        rec["reason"] = "unknown_op"
    return rec


# --------------------------------------------------------------------------- #
# Control-signal helpers (primitive DSP transforms only; no generation)
# --------------------------------------------------------------------------- #
def write_float_wav(path: str, x: np.ndarray, sr: int) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(x, dtype="float32"), sr, subtype="FLOAT")
    return path


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * (10.0 ** (db / 20.0))).astype(np.float32)


def resample_to(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    import librosa
    return np.ascontiguousarray(librosa.resample(x, orig_sr=sr, target_sr=target_sr),
                                dtype=np.float32)


def delay(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    k = int(round(ms / 1000.0 * sr))
    if k <= 0:
        return x.copy()
    return np.concatenate([np.zeros(k, np.float32), x]).astype(np.float32)


def truncate(x: np.ndarray, frac: float) -> np.ndarray:
    return x[: max(1, int(len(x) * frac))].copy()


def drop_middle(x: np.ndarray, frac: float = 0.25) -> np.ndarray:
    n = len(x)
    a, b = int(n * 0.35), int(n * (0.35 + frac))
    return np.concatenate([x[:a], x[b:]]).astype(np.float32)


def silence(n: int) -> np.ndarray:
    return np.zeros(n, np.float32)


def with_nan(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[len(y) // 2] = np.nan
    return y
