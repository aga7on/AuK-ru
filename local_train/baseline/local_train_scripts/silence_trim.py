"""Silence trim: cap lead/tail and shorten over-long internal pauses (crossfaded)."""
import argparse
import os

import numpy as np
import soundfile as sf

SR = 24000


def rms_frames(y, frame=600, hop=240):
    n = 1 + max(0, (len(y) - frame) // hop)
    if n <= 0:
        return np.zeros(1)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx]
    return np.sqrt((fr ** 2).mean(axis=1) + 1e-12)


def find_pauses(y):
    r = rms_frames(y)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    hop_s = 240 / SR
    runs = []
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            runs.append((start, i, (i - start) * hop_s))
            start = None
    if start is not None:
        runs.append((start, len(speech), (len(speech) - start) * hop_s))
    return runs, hop_s


def trim(y, lead_max=0.15, tail_max=0.20, int_cap=0.35, fade_ms=10):
    runs, _ = find_pauses(y)
    if not runs:
        return y, []
    changes = []
    keep_mask = np.ones(len(y), dtype=bool)
    fade = int(SR * fade_ms / 1000)
    for start, end, dur in runs:
        s0, s1 = start * 240, min(len(y), end * 240 + 600)
        is_lead = start == 0
        is_tail = end >= len(find_pauses(y)[0]) and s1 >= len(y) - 700
        if is_lead:
            cut = max(0.0, dur - lead_max)
            if cut > 0:
                n_cut = int(cut * SR)
                keep_mask[s0:s0 + n_cut] = False
                changes.append(("lead", round(dur, 2), round(lead_max, 2)))
        elif s1 >= len(y) - 700:
            cut = max(0.0, dur - tail_max)
            if cut > 0:
                n_cut = int(cut * SR)
                keep_mask[s1 - n_cut:s1] = False
                changes.append(("tail", round(dur, 2), round(tail_max, 2)))
        else:
            cut = max(0.0, dur - int_cap)
            if cut > 0:
                mid = (s0 + s1) // 2
                half = int(cut * SR) // 2
                a, b = max(0, mid - half), min(len(y), mid + half)
                keep_mask[a:a + fade] = False
                keep_mask[b - fade:b] = False
                keep_mask[a + fade:b - fade] = False
                changes.append(("int", round(dur, 2), round(int_cap, 2)))
    out = y[keep_mask]
    return out, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = args.files or [os.path.join(args.in_dir, f) for f in os.listdir(args.in_dir) if f.endswith(".wav")]
    for f in files:
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        out, ch = trim(y)
        name = os.path.basename(f)
        dst = os.path.join(args.out_dir, name)
        sf.write(dst, out, sr)
        print(f"{name}: {len(y)/sr:.2f}s -> {len(out)/sr:.2f}s {ch}")


if __name__ == "__main__":
    main()
