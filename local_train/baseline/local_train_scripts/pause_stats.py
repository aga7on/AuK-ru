"""Pause statistics: gen vs tgt across checkpoints (energy-based VAD)."""
import glob
import os
import sys

import numpy as np
import soundfile as sf

SAMPLES = r"G:\AI\AuK\local_train\run_ru_s1\samples"
SR = 24000


def rms_frames(y, frame=600, hop=240):
    n = 1 + max(0, (len(y) - frame) // hop)
    if n <= 0:
        return np.zeros(1)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx]
    return np.sqrt((fr ** 2).mean(axis=1) + 1e-12)


def pause_stats(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    r = rms_frames(y)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    hop_s = 240 / SR
    pauses = []
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            dur = (i - start) * hop_s
            if dur >= 0.15:
                pauses.append((start * hop_s, dur))
            start = None
    if start is not None and (len(speech) - start) * hop_s >= 0.15:
        pauses.append((start * hop_s, (len(speech) - start) * hop_s))
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
    total = len(y) / SR
    speech_dur = speech.sum() * hop_s
    internal = [(p, d) for p, d in pauses if p > lead and p + d < total - tail]
    return {
        "dur": total,
        "speech": speech_dur,
        "lead": lead,
        "tail": tail,
        "n_int": len(internal),
        "int_total": sum(d for _, d in internal),
        "int_max": max((d for _, d in internal), default=0.0),
    }


def main():
    steps = []
    for f in glob.glob(os.path.join(SAMPLES, "update_*_gen.wav")):
        step = int(os.path.basename(f).split("_")[1])
        if step >= 10000:
            steps.append(step)
    steps.sort()
    print(f"{'update':>7} | {'gen: dur spch pauses(int/tot/max) lead/tail':<46} | tgt: int_tot/max lead/tail")
    for step in steps:
        g = os.path.join(SAMPLES, f"update_{step}_gen.wav")
        t = os.path.join(SAMPLES, f"update_{step}_tgt.wav")
        if not (os.path.exists(g) and os.path.exists(t)):
            continue
        sg, st = pause_stats(g), pause_stats(t)
        print(f"{step:>7} | gen {sg['dur']:5.2f}s spch {sg['speech']:5.2f} int {sg['n_int']:2d}/{sg['int_total']:4.2f}s/{sg['int_max']:4.2f}s ld/tr {sg['lead']:.2f}/{sg['tail']:.2f} | tgt int {st['int_total']:4.2f}s/{st['int_max']:4.2f}s ld/tr {st['lead']:.2f}/{st['tail']:.2f}")


if __name__ == "__main__":
    main()
