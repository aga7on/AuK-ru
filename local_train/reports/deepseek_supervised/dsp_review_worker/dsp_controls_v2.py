"""Focused positive/negative DSP controls for the corrected measurement core.

Runs primitive signal transformations (gain/time-stretch/pitch-shift/delay/
truncation/drop/silence/non-finite/SR change) against known expectations and
asserts the OUTCOME, never `valid=true` alone.

Outputs written next to this file: dsp_v2_controls.json (and control WAVs in
controls/). Exit code 0 iff every control matched its expectation.
"""
from __future__ import annotations

import json
import os

import numpy as np

import dsp_core_v2 as C

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL_DIR = os.path.join(HERE, "controls")
NOISY = os.path.join(C.TOOLS_DIR, "noise_in_val_9599fc9ae0deaf01.wav")
CLEAN = os.path.join(C.KYUTAI_CLEAN_DIR, "9599fc9ae0deaf01.wav")


def _base_with_headroom(peak_target: float = 0.25):
    a = C.load_audio(CLEAN)
    x = a.x / max(1e-9, float(np.max(np.abs(a.x)))) * peak_target
    return np.asarray(x, dtype=np.float32), a.sr


def _rec(measured, expect_text, ok, extra=None):
    out = {"expectation": expect_text, "asserted_ok": bool(ok), "measured": measured}
    if extra:
        out.update(extra)
    return out


def run_controls():
    base, sr = _base_with_headroom()
    results = {}

    def wav(name, x):
        return C.write_float_wav(os.path.join(CTRL_DIR, name), x, sr)

    src_head = wav("src_headroom_float.wav", base)

    # ---------- positive controls ----------
    for sign, db, op in (("up", 6.0, "volume_up"), ("down", -6.0, "volume_down")):
        p = wav(f"pos_vol_{sign}_6db.wav", C.gain(base, db))
        m = C.measure_file(op, src_head, p)
        exp = f"{op} on exactly {db:+.0f} dB -> valid and |delta-({db})|<=0.6"
        ok = bool(m.get("valid") and m.get("delta_db") is not None
                  and abs(m["delta_db"] - db) <= 0.6 and m.get("direction_ok"))
        results[f"pos_volume_{sign}_6db"] = _rec(m, exp, ok)

    for rate, op in ((1.1, "speed_up"), (0.9, "speed_down")):
        import librosa
        p = wav(f"pos_speed_{rate}.wav", librosa.effects.time_stretch(base, rate=rate))
        m = C.measure_file(op, src_head, p)
        exp = f"{op} at rate {rate} -> valid, comparable, |rate-{rate}|<=0.05"
        ok = bool(m.get("valid") and m.get("content_comparable")
                  and m.get("rate_proxy") is not None and abs(m["rate_proxy"] - rate) <= 0.05)
        results[f"pos_speed_{rate}"] = _rec(m, exp, ok)

    for st, op in ((2.0, "pitch_up"), (-2.0, "pitch_down")):
        import librosa
        p = wav(f"pos_pitch_{st}.wav", librosa.effects.pitch_shift(base, sr=sr, n_steps=st))
        m = C.measure_file(op, src_head, p)
        exp = f"{op} by {st:+.0f} st -> valid, |semitones-({st})|<=0.8, no octave flag"
        ok = bool(m.get("valid") and m.get("semitones") is not None
                  and abs(m["semitones"] - st) <= 0.8 and not m.get("octave_error"))
        results[f"pos_pitch_{st}"] = _rec(m, exp, ok)

    # perfect denoise: noisy input vs its own clean ground truth (provenance)
    m = C.measure_file("noise_add", NOISY,
                       wav("pos_denoise_clean.wav", C.load_audio(CLEAN).x))
    exp = "clean ground truth as output -> applicable and snr_gain_db>0"
    ok = bool(m.get("valid") and m.get("applicable") and m.get("direction_ok")
              and m.get("snr_gain_db") is not None and m["snr_gain_db"] > 0)
    results["pos_denoise_perfect"] = _rec(m, exp, ok)

    # ---------- negative / robustness controls ----------
    p = wav("neg_vol_silence.wav", C.silence(len(base)))
    m = C.measure_file("volume_up", src_head, p)
    exp = "all-zero output -> invalid, no gain reported"
    ok = bool(not m.get("valid") and m.get("delta_db") is None
              and m.get("reason") in ("output_has_no_active_speech", "nonfinite_audio"))
    results["neg_vol_silence"] = _rec(m, exp, ok)

    p = wav("neg_vol_nonfinite.wav", C.with_nan(base))
    m = C.measure_file("volume_up", src_head, p)
    exp = "NaN output -> invalid nonfinite"
    ok = bool(not m.get("valid") and m.get("reason") == "nonfinite_audio")
    results["neg_vol_nonfinite"] = _rec(m, exp, ok)

    p = wav("neg_vol_unchanged.wav", base)
    m = C.measure_file("volume_up", src_head, p)
    exp = "0 dB change on volume_up -> valid measurement but outside +6 window, no direction"
    ok = bool(m.get("valid") and m.get("within_prospective_window") is False
              and m.get("direction_ok") is False)
    results["neg_vol_unchanged"] = _rec(m, exp, ok)

    p = wav("neg_vol_wrongdirection.wav", C.gain(base, -6.0))
    m = C.measure_file("volume_up", src_head, p)
    exp = "wrong direction (-6 dB measured as volume_up) -> direction_ok False"
    ok = bool(m.get("valid") and m.get("delta_db") is not None
              and m["delta_db"] < 0 and m.get("direction_ok") is False)
    results["neg_vol_wrongdirection"] = _rec(m, exp, ok)

    import librosa
    sped = librosa.effects.time_stretch(base, rate=1.1)
    p = wav("neg_speed_dropped_words.wav", C.drop_middle(sped, 0.25))
    m = C.measure_file("speed_up", src_head, p)
    exp = "speed-up with a dropped middle chunk -> NOT a clean speed pass (invalid or flagged)"
    ok = bool((not m.get("valid")) or (m.get("content_comparable") is False)
              or (m.get("within_prospective_window") is False))
    results["neg_speed_dropped_words"] = _rec(m, exp, ok)

    p = wav("neg_speed_delay_150ms.wav", C.delay(base, sr, 150.0))
    m = C.measure_file("speed_down", src_head, p)
    exp = "150 ms delay is not a speed change -> rate~1.0 (|rate-1|<=0.05)"
    ok = bool(m.get("rate_proxy") is not None and abs(m["rate_proxy"] - 1.0) <= 0.05)
    results["neg_speed_delay"] = _rec(m, exp, ok)

    # denoise on a delayed noisy copy: alignment must recover; no fabricated gain
    noisy = C.load_audio(NOISY)
    p = wav("neg_denoise_delay_150ms.wav", C.delay(noisy.x, noisy.sr, 150.0))
    m = C.measure_file("noise_add", NOISY, p)
    exp = ("delayed noisy copy -> alignment recovers ~150 ms and snr_gain stays "
           "near 0 (no fabricated denoise)")
    ok = bool(m.get("valid") and m.get("align", {}).get("lag_ms") is not None
              and abs(abs(m["align"]["lag_ms"]) - 150.0) <= 40.0
              and m.get("snr_gain_db") is not None and m["snr_gain_db"] < 3.0
              and m.get("direction_ok") is False)
    results["neg_denoise_delay"] = _rec(m, exp, ok)

    # SR mismatch: explicit resample must be flagged, magnitude preserved
    p_in = wav("neg_sr_src_24k.wav", base)
    down = C.resample_to(base, sr, 16000)
    p = C.write_float_wav(os.path.join(CTRL_DIR, "neg_sr_out_16k.wav"), down, 16000)
    m = C.measure_file("volume_up", p_in, p)
    exp = "16 kHz output vs 24 kHz source -> resampled flag, gain ~0 (no silent truncation)"
    ok = bool(m.get("resampled") and m.get("valid") and m.get("delta_db") is not None
              and abs(m["delta_db"]) <= 0.6)
    results["neg_sr_mismatch"] = _rec(m, exp, ok)

    # silence ground truth must never yield a perfect denoise
    pz = C.write_float_wav(os.path.join(CTRL_DIR, "neg_zero_clean.wav"),
                           C.silence(len(noisy.x)), noisy.sr)
    p_out = wav("neg_denoise_out_zero.wav", C.silence(len(noisy.x)))
    m = C.measure_file("noise_add", pz, p_out)
    exp = "zero ground truth/output -> invalid, never counted as perfect denoise"
    ok = bool(not m.get("valid"))
    results["neg_denoise_zero"] = _rec(m, exp, ok)

    return results


def main():
    res = run_controls()
    path = os.path.join(HERE, "dsp_v2_controls.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    n_ok = sum(1 for r in res.values() if r["asserted_ok"])
    for name, r in res.items():
        print(f"{'PASS' if r['asserted_ok'] else 'FAIL'}  {name}: {r['expectation']}")
    print(f"\ncontrols {n_ok}/{len(res)} asserted")
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
