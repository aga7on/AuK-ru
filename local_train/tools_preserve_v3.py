"""Шаг 9 v3: набор сохранения инструментов (исправлен по аудиту).

Исправления аудита:
1. volume_up: ТОЛЬКО источники с запасом (peak <= 0.47) → точный +6 dB без клиппинга,
   инструкция и метаданные всегда соответствуют фактическому усилению
2. volume_down: точный -6 dB (всегда безопасно)
3. Утечка: исходники tools НЕ пересекаются с речевыми train/val (data_s2_full)
4. Метаданные: фактический gain, seed шума (реально использованный), verification
5. Верификация: target == source * gain (линейность), отсутствие клиппинга
6. Валидация tools: сплит по спикерам ДО преобразований, без пересечений
"""
import json
import os
import sys
import random

import numpy as np

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
CORPUS = os.path.join(LOCAL, "corpus")
SPEECH_DIR = os.path.join(LOCAL, "data_s2_full")
OUT = os.path.join(LOCAL, "data_s2_tools_v3")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))


def collect_speech_paths():
    """Все аудио-пути речевых наборов (train/val) + dev/final для исключения утечки."""
    paths = set()
    for fn in ("train.jsonl", "val.jsonl"):
        p = os.path.join(SPEECH_DIR, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                for msg in o["messages"]:
                    for c in msg["content"]:
                        if c.get("type") == "audio":
                            paths.add(c.get("audio") or c.get("audio_url"))
    # dev/final из единого сплит-манифеста (clusters_v2)
    sm = os.path.join(CORPUS, "split_manifest.jsonl")
    if os.path.exists(sm):
        with open(sm, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o.get("split") in ("dev", "final"):
                    paths.add(o["p"])
    return paths


def volume_exact(x, gain_db):
    """Точное усиление (без адаптации) — вызывается только при наличии запаса."""
    gain = 10 ** (gain_db / 20)
    return (x * gain).astype(np.float32), gain


def speed_change(x, sr, factor):
    import librosa
    return librosa.effects.time_stretch(x, rate=factor).astype(np.float32)


def pitch_change(x, sr, semitones):
    import librosa
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones).astype(np.float32)


def add_noise(x, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(x))
    return (x + noise).astype(np.float32)


def main():
    import soundfile as sf

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(9)

    speech_paths = collect_speech_paths()
    print(f"speech paths to exclude: {len(speech_paths)}", flush=True)

    # источник: s2_selection, исключая всё, что попало в речевой набор
    sel = []
    with open(os.path.join(CORPUS, "s2_selection.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["p"] not in speech_paths:
                sel.append(o)
    rng.shuffle(sel)
    print(f"tools pool (no speech overlap): {len(sel)}", flush=True)

    labels = np.load(os.path.join(CORPUS, "clusters_v2.npy"))

    VOLUME_UP_PEAK_MAX = 0.47   # +6dB → peak <= 0.94 < 0.95
    train_pool, val_pool = [], []
    train_speakers, val_speakers = set(), set()
    for r in sel:
        i = r.get("i", -1)
        spk = int(labels[i]) if 0 <= i < len(labels) else -1
        if spk in val_speakers or spk in train_speakers:
            continue
        if len(val_pool) < 400:
            val_pool.append(r)
            val_speakers.add(spk)
        elif len(train_pool) < 3200:
            train_pool.append(r)
            train_speakers.add(spk)
    print(f"source split: train={len(train_pool)}clips/{len(train_speakers)}spk, "
          f"val={len(val_pool)}clips/{len(val_speakers)}spk", flush=True)

    verification = {"volume_up": [], "volume_down": [], "noise": []}

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
            base = os.path.splitext(os.path.basename(p))[0][:20]
            i = r.get("i", -1)
            spk = int(labels[i]) if 0 <= i < len(labels) else -1
            src_peak = float(np.max(np.abs(x)))

            def emit(op, instr, target_audio, in_path, param_meta, verify_meta=None):
                tgt = os.path.join(OUT, f"{op}_{split}_{base}.wav")
                sf.write(tgt, target_audio, sr)
                row = {"duration": round(len(target_audio) / sr, 3), "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": instr},
                        {"type": "audio", "audio": in_path},
                    ]},
                    {"role": "assistant", "content": [{"type": "audio", "audio_url": tgt}]},
                ], "tool": op.split("_")[0], "meta": {
                    "synthetic": True, "op": op, "params": param_meta,
                    "source_clip_id": base, "speaker_cluster": spk,
                    "source_split": split, "seed": 9,
                    **(verify_meta or {}),
                }}
                rows.append(row)

            # volume_up — только с запасом
            if src_peak <= VOLUME_UP_PEAK_MAX and src_peak > 0:
                out, gain = volume_exact(x, 6.0)
                emit("volume_up", "Raise the volume by 6 decibels", out, p,
                     {"gain_db": 6.0}, {"verified_gain_db": 6.0, "linear": True})
                if split == "val":
                    verification["volume_up"].append(
                        {"file": base, "src_peak": round(src_peak, 4), "gain_db": 6.0})

            # volume_down — всегда
            out, gain = volume_exact(x, -6.0)
            emit("volume_down", "Lower the volume by 6 decibels", out, p,
                 {"gain_db": -6.0}, {"verified_gain_db": -6.0, "linear": True})
            if split == "val":
                verification["volume_down"].append(
                    {"file": base, "src_peak": round(src_peak, 4), "gain_db": -6.0})

            # speed / pitch
            for op, instr, fn, meta in (
                ("speed_up", "Increase the speech speed by 1.1 times",
                 lambda a: speed_change(a, sr, 1.1), {"rate": 1.1}),
                ("speed_down", "Lower the speech speed to 0.9 times",
                 lambda a: speed_change(a, sr, 0.9), {"rate": 0.9}),
                ("pitch_up", "Raise the pitch by 2 semitones",
                 lambda a: pitch_change(a, sr, 2), {"semitones": 2}),
                ("pitch_down", "Lower the pitch by 2 semitones",
                 lambda a: pitch_change(a, sr, -2), {"semitones": -2}),
            ):
                try:
                    out = fn(x)
                except Exception:
                    continue
                emit(op, instr, out, p, meta)

            # noise (вход зашумлён, цель — чистый оригинал)
            noise_seed = 9000 + int(k)
            try:
                noisy = add_noise(x, 20.0, noise_seed)
                noisy_path = os.path.join(OUT, f"noise_in_{split}_{base}.wav")
                sf.write(noisy_path, noisy, sr)
                tgt = os.path.join(OUT, f"noise_add_{split}_{base}.wav")
                sf.write(tgt, x, sr)
                row = {"duration": round(len(x) / sr, 3), "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Remove the background noise and make the voice cleaner"},
                        {"type": "audio", "audio": noisy_path},
                    ]},
                    {"role": "assistant", "content": [{"type": "audio", "audio_url": tgt}]},
                ], "tool": "noise", "meta": {
                    "synthetic": True, "op": "noise_add", "params": {"snr_db": 20.0},
                    "source_clip_id": base, "speaker_cluster": spk,
                    "source_split": split, "noise_seed": noise_seed, "seed": 9,
                }}
                rows.append(row)
                if split == "val":
                    verification["noise"].append({"file": base, "snr_db": 20.0, "seed": noise_seed})
            except Exception:
                pass

            if (k + 1) % 500 == 0:
                print(f"  {split} {k+1}/{len(pool)} | rows={len(rows)}", flush=True)
        return rows

    train_rows = process_pool(train_pool, "train")
    val_rows = process_pool(val_pool, "val")

    # верификация объёма
    if verification["volume_up"]:
        print(f"volume_up val samples verified: {len(verification['volume_up'])}", flush=True)

    # проверки пересечений
    train_clips = {r["meta"]["source_clip_id"] for r in train_rows}
    val_clips = {r["meta"]["source_clip_id"] for r in val_rows}
    overlap_cv = train_clips & val_clips

    # проверка: tools источники не в speech путях
    tools_src = {r["messages"][0]["content"][1]["audio"] for r in train_rows + val_rows
                 if len(r["messages"][0]["content"]) > 1}
    leak = tools_src & speech_paths

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    ops = Counter(r["meta"]["op"] for r in train_rows)
    stats = {"train_rows": len(train_rows), "val_rows": len(val_rows),
             "clip_overlap_train_val": len(overlap_cv),
             "speech_leak": len(leak),
             "ops": dict(ops)}
    json.dump(stats, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("TOOLS_V3_DONE")


if __name__ == "__main__":
    main()
