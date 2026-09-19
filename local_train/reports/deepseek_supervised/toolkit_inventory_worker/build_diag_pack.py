"""Build + validate the authorized 9-WAV diagnostic pack (3 fixtures x 3 controls).

Validation performed here (before any synthesis):
- every source path exists; sha256 + duration + sr + channels recorded;
- source has speech-like energy (active-frame check) and duration within 0.3-30 s;
- ref semantics: pitch/zero-shot fixtures require a valid ref; instruct-tts has none;
- explicit gen_seconds is set for every fixture (for pitch it equals the source length);
- control checkpoint + config existence; full sha256 cross-checked against full_sha256.json.

Writes diag_pack.json in this folder. No model is loaded here.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
AUK = r"G:\AI\AuK"
FULL = json.load(open(os.path.join(HERE, "..", "full_sha256.json"), encoding="utf-8"))
PACK = os.path.join(HERE, "diag_pack.json")
ROOT = os.path.join(HERE, "diag_out")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_check(path: str) -> dict:
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    # active-frame ratio (speech-like energy)
    win = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    n = max(1, (len(x) - win) // hop + 1)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    fr = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(fr + 1e-9)
    active = float(np.mean(db > (db.max() - 35.0)))
    return {"sr": int(sr), "channels": int(info.channels), "frames": int(len(x)),
            "duration_s": round(dur, 4), "rms": round(rms, 5),
            "active_frame_ratio": round(active, 4), "sha256": sha256(path)}


FIXTURES = [
    {
        "id": "diag_pitch_en_up2",
        "task": "pitch_edit",
        "language": "en",
        "instruction": "Raise the pitch by 2 semitones.",
        "ref": os.path.join(AUK, "assets", "demo-input-audio", "pitch", "pitch-1-input.wav"),
        "ref_required": True,
        "gen_seconds": None,  # filled = source duration, validated equality
        "gen_seconds_equals_source": True,
        "expected": "voiced F0 ratio ~ +2.0 semitones; content preserved",
    },
    {
        "id": "diag_zeroshot_en",
        "task": "zero_shot_tts",
        "language": "en",
        "instruction": ("Say the following with the same voice: 'The northern lights shimmered "
                        "above the frozen lake while we watched in silence.'"),
        "ref": os.path.join(AUK, "assets", "demo-input-audio", "zero-shot-tts", "ref.wav"),
        "ref_required": True,
        "gen_seconds": 7.0,
        "gen_seconds_equals_source": False,
        "expected": "intelligible target text in ref voice; sufficient duration; no ref-word echo",
    },
    {
        "id": "diag_instructtts_en_noref",
        "task": "instruct_tts",
        "language": "en",
        "instruction": ("Based on the following description: \"A calm, clear adult female narrator "
                        "with a warm, measured delivery and a slight smile in her voice, speaking at a "
                        "natural pace with crisp articulation.\", generate speech content \"The ancient "
                        "clockmaker wound the brass key three times before sunrise.\"."),
        "ref": None,
        "ref_required": False,
        "gen_seconds": 5.0,
        "gen_seconds_equals_source": False,
        "expected": "intelligible English; calm clear female narrator; ~4.5-5.0 s; no-reference path",
    },
]

CONTROLS = {
    "upstream_base": {
        "ckpt": os.path.join(AUK, "ckpts", "AuK", "auk_base.safetensors"),
        "config": os.path.join(AUK, "ckpts", "AuK", "config.yaml"),
        "sha256_full_expected": FULL["upstream_base"]["sha256"],
    },
    "s1_u10000": {
        "ckpt": os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_10000.safetensors"),
        "config": os.path.join(AUK, "local_train", "run_ru_s1", "merged", "config.yaml"),
        "sha256_full_expected": FULL["s1_merged_10000"]["sha256"],
    },
    "s2_B500": {
        "ckpt": os.path.join(AUK, "local_train", "run_s2_B", "merged", "auk_s2_B_500.safetensors"),
        "config": os.path.join(AUK, "local_train", "run_s2_B", "merged", "config.yaml"),
        "sha256_full_expected": FULL["s2_B500"]["sha256"],
    },
}


def main():
    errors = []
    fixtures = []
    for fx in FIXTURES:
        rec = {
            "id": fx["id"], "task": fx["task"], "language": fx["language"],
            "instruction": fx["instruction"], "ref": fx["ref"],
            "ref_required": fx["ref_required"], "gen_seconds": fx["gen_seconds"],
            "expected": fx["expected"],
        }
        if fx["ref"] is None:
            if fx["ref_required"]:
                errors.append(f"{fx['id']}: ref required but missing")
            if not fx["gen_seconds"]:
                errors.append(f"{fx['id']}: no-ref instruct_tts needs explicit gen_seconds")
            rec["source"] = None
        else:
            if not os.path.isfile(fx["ref"]):
                errors.append(f"{fx['id']}: ref missing {fx['ref']}")
                fixtures.append(rec)
                continue
            s = wav_check(fx["ref"])
            rec["source"] = s
            if not (0.3 <= s["duration_s"] <= 30.0):
                errors.append(f"{fx['id']}: duration {s['duration_s']} outside 0.3-30 s")
            if s["active_frame_ratio"] < 0.15 or s["rms"] < 1e-3:
                errors.append(f"{fx['id']}: source does not look like speech "
                              f"(active={s['active_frame_ratio']}, rms={s['rms']})")
            if fx["gen_seconds_equals_source"]:
                rec["gen_seconds"] = s["duration_s"]
                if fx["gen_seconds"] is not None and abs(s["duration_s"] - fx["gen_seconds"]) > 1e-6:
                    errors.append(f"{fx['id']}: gen_seconds must equal source duration "
                                  f"{s['duration_s']}")
            if fx["id"] == "diag_zeroshot_en" and s["duration_s"] < 5.0:
                errors.append(f"{fx['id']}: ref too short for a 13-word target")
        fixtures.append(rec)

    controls = {}
    for name, c in CONTROLS.items():
        ok = os.path.isfile(c["ckpt"]) and os.path.isfile(c["config"])
        if not ok:
            errors.append(f"control {name}: ckpt/config missing")
        controls[name] = {
            "ckpt": c["ckpt"], "config": c["config"],
            "sha256_full_expected": c["sha256_full_expected"],
            "exists": bool(ok),
        }

    code_files = ["build_diag_pack.py", "run_diag.py"]
    code_hashes = {}
    for cf in code_files:
        p = os.path.join(HERE, cf)
        if os.path.isfile(p):
            code_hashes[cf] = sha256(p)

    pack = {
        "schema": "diag-1",
        "created_by": "toolkit_inventory_worker",
        "authorization": "root-authorized small synthesis diagnostic; 3 fixtures x 3 controls = 9 WAVs",
        "run_config": {"seed": 1234, "nfe": 64, "cfg_strength": 2.0, "trim": False,
                       "bestofn": 1, "dtype": "bf16", "cpu_offload": True,
                       "faithful": "no clipping normalization, no silence trim",
                       "one_engine_at_a_time": True},
        "output_root": ROOT,
        "fixtures": fixtures,
        "controls": controls,
        "code_sha256": code_hashes,
        "validation_errors": errors,
    }
    with open(PACK, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=1)

    print(f"wrote {PACK}")
    print("fixtures:", len(fixtures), "controls:", len(controls), "errors:", len(errors))
    for e in errors:
        print("ERROR:", e)
    for fx in fixtures:
        src = fx.get("source") or {}
        print(f"  {fx['id']}: ref={'yes' if fx['ref'] else 'no'} "
              f"dur={src.get('duration_s')} sr={src.get('sr')} "
              f"active={src.get('active_frame_ratio')} gen_seconds={fx['gen_seconds']}")


if __name__ == "__main__":
    main()
