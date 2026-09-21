# -*- coding: utf-8 -*-
"""Объективная проверка релизных семплов: пол (f0), дубликаты, работает ли эмоция.

Зачем: имена файлов обещают female/male и конкретную эмоцию. Проверяем, правда ли это:
  1. sha256 — нет ли дубликатов (одинаковые размеры попарно подозрительны);
  2. f0 median / range / duration — пол по f0 (мужской ~85–165 Гц, женский ~165–255);
  3. ΔRMS и Δf0 между «эмоциями» одного голоса — меняется ли вообще подача.

usage: python check_samples_honesty.py [--dir local_train/tmp_samples_check/samples]
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"G:\AI\AuK\local_train\tmp_samples_check\samples")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".wav"))
    out = []
    for fn in files:
        p = os.path.join(args.dir, fn)
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        h = sha(p)
        # f0 через простой автокорреляционный пич-трекер на фреймах
        frame = int(0.04 * sr)
        hop = int(0.02 * sr)
        f0s = []
        for i in range(0, max(1, len(x) - frame), hop):
            seg = x[i:i + frame]
            if np.sqrt(np.mean(seg ** 2)) < 0.01:
                continue
            seg = seg - seg.mean()
            ac = np.correlate(seg, seg, "full")[frame - 1:]
            lo, hi = int(sr / 350), int(sr / 70)
            if hi >= len(ac):
                continue
            lag = lo + int(np.argmax(ac[lo:hi]))
            if ac[lag] > 0.3 * ac[0]:
                f0s.append(sr / lag)
        rms = float(np.sqrt(np.mean(x ** 2)))
        out.append({
            "file": fn, "sha256": h[:16], "size": os.path.getsize(p),
            "dur_s": round(len(x) / sr, 2),
            "f0_median": round(float(np.median(f0s)), 1) if f0s else None,
            "f0_range": round(float(np.max(f0s) - np.min(f0s)), 1) if f0s else None,
            "n_voiced_frames": len(f0s), "rms": round(rms, 4),
            "gender_by_f0": (None if not f0s else
                             ("male" if np.median(f0s) < 165 else "female")),
        })

    print("file | sha16 | dur | f0_med | f0_range | rms | gender_by_f0")
    for r in out:
        print(f"{r['file']:28s} {r['sha256']} {r['dur_s']:5.2f}s "
              f"f0={r['f0_median']} range={r['f0_range']} rms={r['rms']} -> {r['gender_by_f0']}")

    shas = [r["sha256"] for r in out]
    print("unique sha:", len(set(shas)), "of", len(shas),
          "| duplicates:", len(shas) - len(set(shas)))

    # дельта между «эмоциями» одного голоса
    males = [r for r in out if r["gender_by_f0"] == "male"]
    fems = [r for r in out if r["gender_by_f0"] == "female"]
    print("gender split:", len(males), "male,", len(fems), "female")
    if len(males) >= 2:
        rs = [r["rms"] for r in males if r["rms"]]
        f0 = [r["f0_median"] for r in males if r["f0_median"]]
        print("male group: rms min %.4f max %.4f (Δ %.1f dB) | f0 min %.0f max %.0f (Δ %.0f Hz)"
              % (min(rs), max(rs), 20 * np.log10(max(rs) / max(min(rs), 1e-9)), min(f0), max(f0),
                 max(f0) - min(f0)))

    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
