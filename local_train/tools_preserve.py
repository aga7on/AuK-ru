"""Шаг 9: набор сохранения инструментов (tool preservation).

Из отобранных клипов создаём синтетические пары:
- volume: запись и версия с известным усилением (+6dB) без клиппинга
- speed: запись и версия с известным изменением скорости (0.9x / 1.1x)
- noise: чистая запись + шум SNR 20dB; цель — чистый оригинал
- pitch: запись и версия с изменением тона (+/-2 полутона) (librosa)

Все примеры помечены как synthetic в meta.
"""
import json
import os
import sys
import random

import numpy as np

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
CORPUS = os.path.join(LOCAL, "corpus")
OUT = os.path.join(LOCAL, "data_s2_tools")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))

TEMPLATES = {
    "volume": "Change the volume of the speech: '{t}'",
    "speed": "Change the speech speed: '{t}'",
    "noise": "Remove the background noise and make the voice cleaner",
    "pitch": "Change the pitch of the speech: '{t}'",
}


def make_volume(x, sr, gain_db=6.0):
    gain = 10 ** (gain_db / 20)
    out = x * gain
    peak = np.max(np.abs(out))
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def make_speed(x, sr, factor=1.1):
    import librosa
    return librosa.effects.time_stretch(x, rate=factor)


def make_noise(x, sr, snr_db=20):
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(42).normal(0, np.sqrt(noise_power), len(x))
    return (x + noise).astype(np.float32)


def make_pitch(x, sr, semitones=2):
    import librosa
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones)


def main():
    import soundfile as sf
    import torchaudio

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(9)
    sel = []
    with open(os.path.join(CORPUS, "s2_selection.jsonl"), encoding="utf-8") as f:
        for line in f:
            sel.append(json.loads(line))
    rng.shuffle(sel)
    sel = sel[:5000]
    print(f"pool: {len(sel)} clips", flush=True)

    rows = []
    for k, r in enumerate(sel):
        p = r["p"]
        if not os.path.exists(p):
            continue
        try:
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if len(x) < sr:
                continue
        except Exception:
            continue
        t = r.get("t", "")[:200]
        base_name = os.path.splitext(os.path.basename(p))[0]

        for op in ("volume", "speed", "pitch"):
            try:
                if op == "volume":
                    modified = make_volume(x, sr)
                    tmpl = TEMPLATES["volume"]
                elif op == "speed":
                    modified = make_speed(x, sr, factor=1.1)
                    tmpl = TEMPLATES["speed"]
                elif op == "pitch":
                    modified = make_pitch(x, sr, semitones=2)
                    tmpl = TEMPLATES["pitch"]
            except Exception:
                continue

            out_path = os.path.join(OUT, f"tool_{op}_{base_name[:20]}.wav")
            sf.write(out_path, modified, sr)

            # для volume/pitch/speed: вход = оригинал (аудио), цель = изменённая версия
            instr = tmpl.format(t=t)
            row = {"duration": round(len(modified) / sr, 3), "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": instr},
                    {"type": "audio", "audio": p},
                ]},
                {"role": "assistant", "content": [{"type": "audio", "audio_url": out_path}]},
            ], "tool": op, "meta": {"synthetic": True, "op": op}}
            rows.append(row)

            # шум: вход = зашумлённая версия, цель = чистый оригинал
            if op == "noise" and k < 1000:
                noisy = make_noise(x, sr)
                noisy_path = os.path.join(OUT, f"tool_noise_{base_name[:20]}.wav")
                sf.write(noisy_path := os.path.join(OUT, f"tool_noise_in_{base_name[:20]}.wav"), noisy, sr)
                row2 = {"duration": round(len(x) / sr, 3), "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": TEMPLATES["noise"]},
                        {"type": "audio", "audio": noisy_path},
                    ]},
                    {"role": "assistant", "content": [{"type": "audio", "audio_url": p}]},
                ], "tool": "noise", "meta": {"synthetic": True, "op": "noise_denoise"}}
                rows.append(row2)

        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(sel)} | rows: {len(rows)}", flush=True)

    # разделить train/val
    rng.shuffle(rows)
    val_n = max(1, len(rows) // 20)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[val_n:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[:val_n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TOOLS_DONE: {len(rows)} rows (train {len(rows)-val_n}, val {val_n})")


if __name__ == "__main__":
    main()
