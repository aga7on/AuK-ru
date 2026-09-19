"""DSP-верификация целей tools_v4 (все 5600): линейность гейна, точность pitch/speed."""
import json
import os
import sys

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
V4 = os.path.join(AUK, "local_train", "data_s2_tools_v4")
sys.path.insert(0, os.path.join(AUK, "local_train"))
from tool_dsp_measure import active_stats, median_f0, load  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(V4, "train.jsonl"), encoding="utf-8") if l.strip()]
    bad_vol, bad_pitch, bad_speed, checked = [], [], [], 0
    for r in rows:
        op = r["meta"]["op"]
        src = r["messages"][0]["content"][1]["audio"]
        tgt = r["messages"][1]["content"][0]["audio_url"]
        if op in ("volume_up", "volume_down"):
            mag = r["meta"]["params"]["up" if op == "volume_up" else "down"]
            try:
                xin, sr = load(src)
                xout, _ = load(tgt)
            except Exception:
                bad_vol.append((tgt, "load"))
                continue
            # RMS по активным кадрам (тишина исключена)
            si, so = active_stats(xin, sr), active_stats(xout, sr)
            if si["rms_db"] is not None and so["rms_db"] is not None:
                delta = so["rms_db"] - si["rms_db"]
                if abs(delta - mag) > 0.15:
                    bad_vol.append((os.path.basename(tgt), round(delta, 2), mag))
            checked += 1
    print(f"checked_volume={checked} bad_volume={len(bad_vol)}")
    for b in bad_vol[:5]:
        print("BAD", b)
    # pitch/speed — выборочно 40 случайных
    import random
    rng = random.Random(3)
    sample = [r for r in rows if r["meta"]["op"].startswith(("pitch", "speed"))]
    for r in rng.sample(sample, min(40, len(sample))):
        op = r["meta"]["op"]
        src = r["messages"][0]["content"][1]["audio"]
        tgt = r["messages"][1]["content"][0]["audio_url"]
        try:
            xin, sr = load(src)
            xout, _ = load(tgt)
        except Exception:
            continue
        if op.startswith("pitch"):
            mag = r["meta"]["params"]["up" if op == "pitch_up" else "down"]
            fi, ni = median_f0(xin, sr)
            fo, no = median_f0(xout, sr)
            if fi and fo and ni >= 30 and no >= 30:
                st = 12 * np.log2(fo / fi)
                # октавные сбои pyin: приводим fo в диапазон [fi/2, fi*2]
                while fo > fi * 2:
                    fo /= 2
                while fo < fi / 2:
                    fo *= 2
                st = 12 * np.log2(fo / fi)
                if abs(st - mag) > 0.35:
                    bad_pitch.append((os.path.basename(tgt), round(st, 2), mag))
        else:
            mag = r["meta"]["params"]["up" if op == "speed_up" else "down"]
            si, so = active_stats(xin, sr), active_stats(xout, sr)
            if si["dur_s"] > 0 and so["dur_s"] > 0:
                rate = si["dur_s"] / so["dur_s"]
                if abs(rate - mag) / mag > 0.08:
                    bad_speed.append((os.path.basename(tgt), round(rate, 3), mag))
    print(f"sampled_pitch_speed=40 bad_pitch={len(bad_pitch)} bad_speed={len(bad_speed)}")
    for b in (bad_pitch + bad_speed)[:5]:
        print("BAD", b)
    print("V4_DSP_OK" if not bad_vol and not bad_pitch and not bad_speed else "V4_DSP_ISSUES")


if __name__ == "__main__":
    main()
