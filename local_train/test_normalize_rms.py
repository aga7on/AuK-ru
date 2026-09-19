"""Тест normalize_rms: спред громкости сидов должен сжаться до ~0, тихие не amplify."""
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, r"G:\AI\AuK\src")

from auk.infer.quality import normalize_rms


def rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)


def rms_db_active(x, sr=24000):
    # по маске речи, как в normalize_rms
    frame, hop = 600, 240
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0:
        return rms_db(x)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(r + 1e-9)
    mask = db > (db.max() - 35.0)
    sel = np.concatenate([np.arange(i * hop, min(len(x), i * hop + frame)) for i in np.flatnonzero(mask)])
    return rms_db(x[np.unique(sel)])


before, after = [], []
import glob
files = sorted(glob.glob(r"G:\AI\AuK\local_tests\tmp_seed_probe\clone0*_s*.wav"))[:12]
for f in files:
    a, sr = sf.read(f)
    t = torch.from_numpy(a).float()
    out = normalize_rms(t).numpy()
    b, af = rms_db_active(a), rms_db_active(out)
    peak = np.abs(out).max()
    before.append(b)
    after.append(af)
    print(f.split("\\")[-1], "before", round(b, 1), "-> after", round(af, 1), "peak", round(float(peak), 3))

print("spread before: max-min", round(max(before) - min(before), 1), "dB")
print("spread after:  max-min", round(max(after) - min(after), 1), "dB")
assert max(after) - min(after) < 2.0, "spread not compressed"
assert min(after) > -22.5, "some sample still too quiet"
assert all(np.abs(out).max() <= 0.951 for out in [normalize_rms(torch.from_numpy(sf.read(f)[0]).float()).numpy() for f in files]), "peak ceiling violated"
print("PASS")
