"""Objective DSP measurement of the seven tool operations across the five s2 variants.

Measures, per candidate file vs its source, independent of Gemini:
- volume_up/down : speech-active RMS delta (dB), peak/clipping ratio;
- speed_up/down  : active-speech duration ratio (rate), not file length;
- pitch_up/down  : median voiced F0 ratio -> semitones (librosa.pyin);
- noise_add      : denoise, using the KNOWN clean source: segmental SNR improvement and
                   speech-preservation correlation (noise_add = "remove background noise").
Invalid measurements are flagged, never counted as pass. Positive/negative DSP controls
validate the pipeline itself. Writes JSON + a markdown table.
"""
import json
import os

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
OUTDIR = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
CLEAN_DIR = r"G:\AI\kyutai-ru\data\ru_wav_mfa"
VARIANTS = ["u10000", "A@250", "A@500", "B@250", "B@500", "s3@2000", "s4@4000", "s5@6000", "s5@4500", "s6@5000", "s6@6000", "s7@5750"]
DIRS = {"u10000": "u0_control", "A@250": "s2_A_250", "A@500": "s2_A_500",
        "B@250": "s2_B_250", "B@500": "s2_B_500", "s3@2000": "s3_control",
        "s4@4000": "s4_control", "s5@6000": "s5_control", "s5@4500": "s5_4500_control",
        "s6@5000": "s6_5000_control", "s6@6000": "s6_control", "s7@5750": "s7_5750_control"}
OPS = ["volume_up", "volume_down", "speed_up", "speed_down", "pitch_up", "pitch_down", "noise_add"]
N_PER_OP = 5


def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x), sr


def frame_rms(x, win, hop):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def speech_mask(x, sr, frame=0.025, hop=0.010, rel_db=30.0):
    win, hp = int(frame * sr), int(hop * sr)
    rms = frame_rms(x, win, hp)
    db = 20 * np.log10(rms + 1e-9)
    return db > (db.max() - rel_db), hp


def active_stats(x, sr):
    mask, hp = speech_mask(x, sr)
    if not np.any(mask):
        return {"dur_s": 0.0, "rms_db": None, "n_active": 0}
    rms = frame_rms(x, int(0.025 * sr), hp)[: len(mask)]
    act = rms[mask]
    return {"dur_s": float(mask.sum() * hp / sr),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-9)),
            "n_active": int(mask.sum())}


def clip_ratio(x):
    return float(np.mean(np.abs(x) >= 0.985)) if x.size else 0.0


def median_f0(x, sr):
    import librosa
    try:
        # trim to active speech: pyin assigns spurious ~fmin floors to unvoiced
        # leading segments, which skew the median (proven on ref 218abd71759f0df1).
        m, _ = speech_mask(x, sr)
        if m is not None and m.sum() > 3:
            win, hp = int(0.025 * sr), int(0.010 * sr)
            idx = np.where(m)[0]
            s, e = max(0, idx[0] * hp - win), min(len(x), (idx[-1] + 1) * hp + win)
            if e - s > int(0.3 * sr):
                x = x[s:e]
        f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr, frame_length=2048)
        v = f0[voiced & np.isfinite(f0)]
        # drop fmin-floor frames (unvoiced noise reported at ~60 Hz)
        v = v[v > 65]
        if v.size < 10:
            return None, int(v.size)
        return float(np.median(v)), int(v.size)
    except Exception:
        return None, 0


def segmental_snr(ref, est, sr, frame=0.025, rel_db=35.0):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    win, hp = int(frame * sr), int(frame * sr // 2)
    rr = frame_rms(ref, win, hp)
    ee = frame_rms(est, win, hp)
    m = len(rr)
    ref2, est2 = ref[:m * hp + win], est[:m * hp + win]
    fr = np.lib.stride_tricks.sliding_window_view(ref2, win)[::hp]
    fe = np.lib.stride_tricks.sliding_window_view(est2, win)[::hp]
    act = rr > (rr.max() - rel_db) if rr.size else np.zeros(0, bool)
    if act.sum() < 5:
        return None
    sig = np.mean(fr[act] ** 2, axis=1)
    err = np.mean((fe[act] - fr[act]) ** 2, axis=1)
    val = 10 * np.log10((sig + 1e-12) / (err + 1e-12))
    return float(np.mean(val))


def corr_active(a, b, sr):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m, _ = speech_mask(a, sr)
    win, hp = int(0.025 * sr), int(0.010 * sr)
    if m.sum() < 5:
        return None
    step = max(1, win // 2)
    aa, bb = [], []
    for i, active in enumerate(m):
        if active:
            s = i * hp
            aa.append(a[s:s + win]); bb.append(b[s:s + win])
    A, B = np.concatenate(aa), np.concatenate(bb)
    if A.std() < 1e-6 or B.std() < 1e-6:
        return None
    return float(np.corrcoef(A, B)[0, 1])


def clean_id(ref_path):
    b = os.path.basename(ref_path)
    if b.startswith("noise_in_val_"):
        return b[len("noise_in_val_"):-4]
    return b[:-4]


def measure_one(op, src, out_path):
    try:
        xin, sr = load(src)
        xout, sr2 = load(out_path)
    except Exception as e:
        return {"valid": False, "reason": f"load: {type(e).__name__}"}
    if len(xin) == 0 or len(xout) == 0:
        return {"valid": False, "reason": "empty audio"}
    si, so = active_stats(xin, sr), active_stats(xout, sr2)
    r = {"valid": False, "sr_in": sr, "dur_in": round(len(xin) / sr, 3),
         "dur_out": round(len(xout) / sr2, 3), "active_in_s": round(si["dur_s"], 3),
         "active_out_s": round(so["dur_s"], 3)}
    if si["n_active"] < 5:
        r["reason"] = "source has no active speech"
        return r
    if op in ("volume_up", "volume_down"):
        if so["rms_db"] is None:
            r["reason"] = "output has no active speech"
            return r
        r.update(delta_db=round(so["rms_db"] - si["rms_db"], 2),
                 clip_in=round(clip_ratio(xin), 5), clip_out=round(clip_ratio(xout), 5),
                 valid=True)
    elif op in ("speed_up", "speed_down"):
        if so["dur_s"] < 0.05:
            r["reason"] = "output has no active speech"
            return r
        r.update(rate=round(si["dur_s"] / so["dur_s"], 3), valid=True)
    elif op in ("pitch_up", "pitch_down"):
        fi, ni = median_f0(xin, sr)
        fo, no = median_f0(xout, sr2)
        # octave-fold: bring both medians into the same octave band [fi/2, fi*2]
        if fi and fo:
            while fo < fi / 2:
                fo *= 2.0
            while fo > fi * 2:
                fo /= 2.0
        if fi and fo and ni >= 10 and no >= 10:
            r.update(f0_in=round(fi, 1), f0_out=round(fo, 1),
                     semitones=round(12 * np.log2(fo / fi), 2), voiced_in=ni, voiced_out=no, valid=True)
        else:
            r["reason"] = f"insufficient voiced frames (in={ni}, out={no})"
    elif op == "noise_add":
        cid = clean_id(src)
        cp = os.path.join(CLEAN_DIR, cid + ".wav")
        if not os.path.exists(cp):
            r["reason"] = "clean source not found"
            return r
        xc, _ = load(cp)
        snr_in = segmental_snr(xc, xin, sr)
        snr_out = segmental_snr(xc, xout, sr2)
        cor = corr_active(xc, xout, sr)
        if snr_in is None or snr_out is None:
            r["reason"] = "no active frames for SNR"
            return r
        r.update(snr_in_db=round(snr_in, 2), snr_out_db=round(snr_out, 2),
                 snr_gain_db=round(snr_out - snr_in, 2),
                 preserve_corr=round(cor, 3) if cor is not None else None, valid=True)
    return r


def controls():
    """Validate the DSP pipeline against known manipulations."""
    ctrl = {}
    base, sr = load(os.path.join(LT, "phonetic_control", "update_u10000_gen.wav")) \
        if os.path.exists(os.path.join(LT, "phonetic_control", "update_u10000_gen.wav")) \
        else load(os.path.join(LT, "user_ref.wav"))
    tmp = os.path.join(OUTDIR, "dsp_controls")
    os.makedirs(tmp, exist_ok=True)

    def w(name, x):
        p = os.path.join(tmp, name)
        sf.write(p, x.astype("float32"), sr)
        return p

    for sign, g in (("up", 6.0), ("down", -6.0)):
        p = w(f"ctl_vol_{sign}.wav", base * (10 ** (g / 20)))
        ctrl[f"volume_{sign}_6db"] = measure_one("volume_up" if g > 0 else "volume_down", w("ctl_vol_src.wav", base), p)
    p = w("ctl_speed_110.wav", resample_lin(base, 1.0 / 1.1))
    ctrl["speed_up_110"] = measure_one("speed_up", w("ctl_speed_src.wav", base), p)
    try:
        import librosa
        for st in (2, -2):
            shifted = librosa.effects.pitch_shift(base, sr=sr, n_steps=st)
            p = w(f"ctl_pitch_{st}.wav", shifted)
            ctrl[f"pitch_{st}st"] = measure_one("pitch_up" if st > 0 else "pitch_down", w("ctl_pitch_src.wav", base), p)
    except Exception as e:
        ctrl["pitch"] = {"valid": False, "reason": f"librosa pitch_shift: {e}"}
    rng = np.random.default_rng(0)
    noisy = (base + 0.03 * rng.standard_normal(len(base))).astype("float32")
    snr_in = segmental_snr(base, noisy, sr)
    snr_out = segmental_snr(base, base.astype("float32"), sr)
    ctrl["denoise_known_clean"] = {
        "valid": snr_in is not None and snr_out is not None,
        "snr_in_db": round(snr_in, 2) if snr_in is not None else None,
        "snr_out_db": round(snr_out, 2) if snr_out is not None else None,
        "snr_gain_db": round(snr_out - snr_in, 2) if (snr_in is not None and snr_out is not None) else None,
    }
    return ctrl


def resample_lin(x, factor):
    n = int(len(x) * factor)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype("float32")


def main():
    man = json.load(open(os.path.join(LT, "u0_control", "manifest.json"), encoding="utf-8"))
    refs = {m["id"]: m.get("ref") for m in man if m["kind"] == "tool"}
    results = {}
    for op in OPS:
        results[op] = {}
        for v in VARIANTS:
            rows = []
            for i in range(N_PER_OP):
                tid = f"tool_{op}_{i}"
                src = refs.get(tid)
                out_path = os.path.join(LT, DIRS[v], f"{tid}.wav")
                if not src or not os.path.exists(out_path):
                    rows.append({"file": tid, "valid": False, "reason": "file/ref missing"})
                    continue
                m = measure_one(op, src, out_path)
                m["file"] = tid
                rows.append(m)
            results[op][v] = rows
    ctrl = controls()
    out = {"ops": results, "controls": ctrl}
    json.dump(out, open(os.path.join(OUTDIR, "tool_dsp_measure.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    key_metric = {"volume_up": "delta_db", "volume_down": "delta_db",
                  "speed_up": "rate", "speed_down": "rate",
                  "pitch_up": "semitones", "pitch_down": "semitones",
                  "noise_add": "snr_gain_db"}
    L = ["# Объективный DSP-замер инструментов (16.09.2026)", "",
         "Источник: `tool_dsp_measure.py`, `tool_dsp_measure.json`. Замер по фактическим WAV",
         "против источника из pack. Невалидные замеры помечены и в агрегат не входят.", ""]
    for op in OPS:
        km = key_metric[op]
        L.append(f"## {op} (метрика: {km})")
        L.append("")
        if op == "noise_add":
            L.append("| вариант | n valid/total | snr_gain_db | snr_in | snr_out | preserve_corr |")
        elif op in ("volume_up", "volume_down"):
            L.append("| вариант | n valid/total | delta_db | clip_in | clip_out |")
        else:
            L.append("| вариант | n valid/total | " + km + " |")
        L.append("|---|---:|---:|---:|---:|---:|" if op in ("volume_up", "volume_down", "noise_add")
                 else "|---|---:|---:|")
        for v in VARIANTS:
            rows = results[op][v]
            valid = [r for r in rows if r.get("valid")]
            if op == "noise_add":
                vals = [r.get(km) for r in valid if isinstance(r.get(km), (int, float))]
                si = [r.get("snr_in_db") for r in valid if isinstance(r.get("snr_in_db"), (int, float))]
                so = [r.get("snr_out_db") for r in valid if isinstance(r.get("snr_out_db"), (int, float))]
                pc = [r.get("preserve_corr") for r in valid if isinstance(r.get("preserve_corr"), (int, float))]
                L.append(f"| {v} | {len(valid)}/{len(rows)} | {np.mean(vals):.2f} | {np.mean(si):.2f} | "
                         f"{np.mean(so):.2f} | {np.mean(pc):.2f} |" if vals and si and so and pc else
                         f"| {v} | {len(valid)}/{len(rows)} | - | - | - | - |")
            elif op in ("volume_up", "volume_down"):
                vals = [r.get(km) for r in valid if isinstance(r.get(km), (int, float))]
                ci = [r.get("clip_in") for r in valid if isinstance(r.get("clip_in"), (int, float))]
                co = [r.get("clip_out") for r in valid if isinstance(r.get("clip_out"), (int, float))]
                L.append(f"| {v} | {len(valid)}/{len(rows)} | {np.mean(vals):.2f} | {np.mean(ci):.4f} | "
                         f"{np.mean(co):.4f} |" if vals else f"| {v} | {len(valid)}/{len(rows)} | - | - | - |")
            else:
                vals = [r.get(km) for r in valid if isinstance(r.get(km), (int, float))]
                L.append(f"| {v} | {len(valid)}/{len(rows)} | {np.mean(vals):.3f} |" if vals
                         else f"| {v} | {len(valid)}/{len(rows)} | - |")
        L.append("")
    L.append("## DSP-контроли (проверка самого пайплайна)")
    L.append("")
    L.append("| контроль | metric | значение | valid |")
    L.append("|---|---|---:|---|")
    for name, r in ctrl.items():
        metrics = {k: r.get(k) for k in ("delta_db", "rate", "semitones", "snr_gain_db") if isinstance(r.get(k), (int, float))}
        L.append(f"| {name} | {list(metrics.keys())} | {list(metrics.values())} | {r.get('valid')} |")
    L.append("")
    L.append(f"Ожидания: volume ±6 дБ, speed rate 1.1, pitch ±2 полутона, denoise snr_gain>0.")
    open(os.path.join(OUTDIR, "TOOL_DSP.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written TOOL_DSP.md and tool_dsp_measure.json")
    for name, r in ctrl.items():
        print("CTRL", name, {k: r.get(k) for k in ("delta_db", "rate", "semitones", "snr_gain_db", "valid", "reason")})


if __name__ == "__main__":
    main()
