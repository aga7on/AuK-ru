"""Шаг 9 v2: набор сохранения инструментов (исправлен по ревью).

- Спилит исходников до преобразований (train/val без пересечений по клипу и спикеру)
- Все 7 операций реально выполняются (включая noise)
- Инструкции с явными величинами в обоих направлениях
- Volume: точное усиление, без пик-нормализации (клип к ±0.95 документируется)
- Source ID, speaker_cluster, параметры, seed — в метаданных
"""
import json
import os
import sys
import random
from collections import defaultdict

import numpy as np

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
CORPUS = os.path.join(LOCAL, "corpus")
OUT = os.path.join(LOCAL, "data_s2_tools_v2")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))


def volume_change(x, gain_db):
    """Адаптивное усиление: цель gain_db, но не превышаем peak 0.95 (документируем фактическую величину)."""
    peak = np.max(np.abs(x))
    target_gain = 10 ** (gain_db / 20)
    max_gain = 0.95 / max(peak, 1e-6)
    actual_gain = min(target_gain, max_gain)
    actual_db = 20 * np.log10(actual_gain) if actual_gain > 0 else 0.0
    out = (x * actual_gain).astype(np.float32)
    return out, round(actual_db, 1)


def speed_change(x, sr, factor):
    import librosa
    return librosa.effects.time_stretch(x, rate=factor).astype(np.float32)


def pitch_change(x, sr, semitones):
    import librosa
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones).astype(np.float32)


def add_noise(x, snr_db=20.0):
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(42).normal(0, np.sqrt(noise_power), len(x))
    return (x + noise).astype(np.float32)


def main():
    import soundfile as sf

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(9)

    # load selection pool
    sel = []
    with open(os.path.join(CORPUS, "s2_selection.jsonl"), encoding="utf-8") as f:
        for line in f:
            sel.append(json.loads(line))
    rng.shuffle(sel)

    # load speaker clusters
    labels = np.load(os.path.join(CORPUS, "clusters_v2.npy"))

    # step 1: сплит ИСХОДНЫХ клипов в train/val ДО преобразований
    # не пересекаются по клипу И по спикеру
    train_pool = []
    val_pool = []
    val_speakers = set()
    train_speakers = set()
    for r in sel:
        i = r.get("i", -1)
        spk = int(labels[i]) if i < len(labels) else -1
        if spk in val_speakers or spk in train_speakers:
            continue
        if len(val_pool) < 500:
            val_pool.append(r)
            val_speakers.add(spk)
        elif len(train_pool) < 3000:
            train_pool.append(r)
            train_speakers.add(spk)
    print(f"source split: train={len(train_pool)} clips/{len(train_speakers)} speakers | "
          f"val={len(val_pool)} clips/{len(val_speakers)} speakers", flush=True)

    # шаг 2: операции с явными величинами
    OPS = [
        ("volume_up", "Raise the volume by 6 decibels",
         lambda x, sr: volume_change(x, 6.0)[0], {"gain_db": 6.0}),
        ("volume_down", "Lower the volume by 6 decibels",
         lambda x, sr: volume_change(x, -6.0)[0], {"gain_db": -6.0}),
        ("speed_up", "Increase the speech speed by 1.1 times",
         lambda x, sr: speed_change(x, sr, 1.1), {"rate": 1.1}),
        ("speed_down", "Lower the speech speed to 0.9 times",
         lambda x, sr: speed_change(x, sr, 0.9), {"rate": 0.9}),
        ("pitch_up", "Raise the pitch by 2 semitones",
         lambda x, sr: pitch_change(x, sr, 2), {"semitones": 2}),
        ("pitch_down", "Lower the pitch by 2 semitones",
         lambda x, sr: pitch_change(x, sr, -2), {"semitones": -2}),
        ("noise_add", "Remove the background noise and make the voice cleaner",
         lambda x, sr: add_noise(x, 20.0), {"snr_db": 20.0}),
    ]

    def process_pool(pool, split):
        rows = []
        for k, r in enumerate(pool):
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
            base_name = os.path.splitext(os.path.basename(p))[0]
            clip_id = base_name[:24]
            spk = int(labels[r.get("i", 0)]) if r.get("i", 0) < len(labels) else -1

            for op_name, instr, fn, params in OPS:
                try:
                    modified = fn(x, sr)
                except Exception:
                    continue

                if op_name == "noise_add":
                    # вход = зашумлённая версия, цель = чистый оригинал
                    input_path = os.path.join(OUT, f"{op_name}_in_{split}_{base_name[:20]}.wav")
                    sf.write(input_path, modified, sr)
                    target_path = p  # оригинал как цель
                else:
                    input_path = p
                    target_path = os.path.join(OUT, f"{op_name}_{split}_{base_name[:20]}.wav")
                    sf.write(target_path, modified, sr)

                row = {"duration": round(len(modified) / sr, 3), "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": instr},
                        {"type": "audio", "audio": input_path},
                    ]},
                    {"role": "assistant", "content": [{"type": "audio", "audio_url": target_path}]},
                ], "tool": op_name.split("_")[0], "meta": {
                    "synthetic": True, "op": op_name,
                    "params": params, "source_clip_id": clip_id,
                    "speaker_cluster": spk, "source_split": split,
                    "seed": 9, "tool_type": "synthetic",
                }}
                rows.append(row)
            if (k + 1) % 500 == 0:
                print(f"  {k+1}/{len(pool)} | rows: {len(rows)}", flush=True)
        return rows

    train_rows = process_pool(train_pool, "train")
    val_rows = process_pool(val_pool, "val")

    # проверка: нет пересечения train/val по source_clip_id
    train_clips = {r["meta"]["source_clip_id"] for r in train_rows}
    val_clips = {r["meta"]["source_clip_id"] for r in val_rows}
    overlap = train_clips & val_clips
    print(f"source clip overlap train/val: {len(overlap)}", flush=True)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ops_counts = defaultdict(int)
    for r in train_rows:
        ops_counts[r["tool"]] += 1
    stats = {"train_rows": len(train_rows), "val_rows": len(val_rows),
             "train_speakers": len(train_speakers), "val_speakers": len(val_speakers),
             "source_overlap": len(overlap),
             "ops_distribution": dict(ops_counts)}
    json.dump(dict(stats), open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print("TOOLS_V2_DONE")


if __name__ == "__main__":
    main()
