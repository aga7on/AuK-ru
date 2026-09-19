"""Evaluate training samples (gen vs tgt) with CPU metrics; append to jsonl.

Watches <samples_dir>/update_*_gen.wav, finds the sibling _tgt.wav and logs:
duration, RMS, peak, median F0 (gender proxy) and DNSMOS OVRL for both files.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf


def f0_median(x, sr):
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    if len(x) < frame * 2:
        return 0.0
    n_frames = 1 + (len(x) - frame) // hop
    win = np.hanning(frame).astype(np.float32)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    f0s = []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame] * win
        if np.sqrt(np.mean(seg * seg)) < 1e-3:
            continue
        seg = seg - seg.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * frame)) ** 2)[:max_lag + 1]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        band = ac[min_lag:max_lag + 1]
        if len(band) == 0:
            continue
        if float(band.max()) < 0.35:
            continue
        lag = min_lag + int(np.argmax(band))
        f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def analyze(path, dnsmos_mod):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    wav16 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)),
                      np.arange(len(x)), x).astype(np.float32)
    d = dnsmos_mod.run(wav16, 16000)
    return {
        "dur": round(dur, 2), "rms": round(rms, 4), "f0": round(f0_median(x, sr), 1),
        "ovrl": round(float(d.get("ovrl_mos", 0.0)), 2), "sig": round(float(d.get("sig_mos", 0.0)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", default=r"G:\AI\AuK\local_train\run_ru\samples")
    ap.add_argument("--out", default=r"G:\AI\AuK\local_train\run_ru\eval_samples.jsonl")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    from speechmos import dnsmos as dnsmos_mod

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["gen"])
            except Exception:
                pass
    print(f"watcher started | samples={args.samples_dir} | already evaluated: {len(done)}", flush=True)
    while True:
        for gen in sorted(glob.glob(os.path.join(args.samples_dir, "update_*_gen.wav"))):
            if gen in done:
                continue
            tgt = gen.replace("_gen.wav", "_tgt.wav")
            if not os.path.exists(tgt):
                continue
            try:
                g = analyze(gen, dnsmos_mod)
                t = analyze(tgt, dnsmos_mod)
                update = int(os.path.basename(gen).split("_")[1])
                row = {"update": update, "gen": gen, "tgt": tgt, "gen_metrics": g, "tgt_metrics": t,
                       "dur_ratio": round(g["dur"] / max(t["dur"], 0.01), 2)}
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done.add(gen)
                print(f"[u{update}] gen ovrl={g['ovrl']} f0={g['f0']} dur={g['dur']} | "
                      f"tgt ovrl={t['ovrl']} f0={t['f0']} | ratio={row['dur_ratio']}", flush=True)
            except Exception as exc:
                print(f"eval failed for {gen}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
