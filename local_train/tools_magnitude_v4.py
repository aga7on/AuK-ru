"""Tools v4: magnitude-вариативные цели (починка volume_up/pitch перелёта s3).

Диагноз S3_RESULTS: все примеры tools_v3 имеют ровно +6 дБ / ±2 st — модель не учит
отображение "величина в инструкции -> величина эффекта". v4 даёт разброс:
  volume_up: +3/+6/+9 дБ (только с запасом пика), volume_down: -3/-6/-9 дБ
  pitch_up: +1/+2/+3 st, pitch_down: -1/-2/-3 st
  speed_up: x1.1/x1.2, speed_down: x0.9/x0.8 (пройдено, для закрепления)
Цели точные DSP (линейный гейн / librosa pitch_shift / time_stretch).
Источник: тот же пул (s2_selection минус речь), те же спикеры допустимы — продолжаем s3.
Выход: data_s2_tools_v4/train.jsonl + build_report.json.
"""
import json
import os
import random
import sys

import numpy as np

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
CORPUS = os.path.join(LOCAL, "corpus")
SPEECH_DIR = os.path.join(LOCAL, "data_s2_full")
OUT = os.path.join(LOCAL, "data_s2_tools_v4")
sys.path.insert(0, LOCAL)


def collect_speech_paths():
    paths = set()
    for sub in ("v2_after_identity", "v3_s3_mix"):
        for fn in ("train.jsonl", "val.jsonl"):
            p = os.path.join(SPEECH_DIR, sub, fn)
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                o = json.loads(line)
                for msg in o["messages"]:
                    for c in msg["content"]:
                        if c.get("type") == "audio":
                            paths.add(c.get("audio") or c.get("audio_url"))
    return paths


MAGNITUDES = {
    "volume_up": [(3, "Raise the volume by 3 decibels"),
                  (6, "Raise the volume by 6 decibels"),
                  (9, "Raise the volume by 9 decibels")],
    "volume_down": [(-3, "Lower the volume by 3 decibels"),
                    (-6, "Lower the volume by 6 decibels"),
                    (-9, "Lower the volume by 9 decibels")],
    "pitch_up": [(1, "Raise the pitch by 1 semitone"),
                 (2, "Raise the pitch by 2 semitones"),
                 (3, "Raise the pitch by 3 semitones")],
    "pitch_down": [(-1, "Lower the pitch by 1 semitone"),
                   (-2, "Lower the pitch by 2 semitones"),
                   (-3, "Lower the pitch by 3 semitones")],
    "speed_up": [(1.1, "Increase the speech speed by 1.1 times"),
                 (1.2, "Increase the speech speed by 1.2 times")],
    "speed_down": [(0.9, "Lower the speech speed to 0.9 times"),
                   (0.8, "Lower the speech speed to 0.8 times")],
}

N_PER_OP_MAG = 350  # строк на (операция, величина) -> ~ (3+3+3+3+2+2)*350 = 5600


def main():
    import soundfile as sf
    import librosa

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(11)

    speech_paths = collect_speech_paths()
    sel = []
    for line in open(os.path.join(CORPUS, "s2_selection.jsonl"), encoding="utf-8"):
        o = json.loads(line)
        if o["p"] not in speech_paths:
            sel.append(o)
    rng.shuffle(sel)
    print(f"pool={len(sel)}", flush=True)

    rows = []
    per_mag = {}
    for op, mags in MAGNITUDES.items():
        for mag, instr in mags:
            key = f"{op}@{mag}"
            per_mag[key] = 0

    for r in sel:
        p = r["p"]
        if not os.path.exists(p):
            continue
        # выбрать операции, которым ещё нужны строки
        need = [k for k, v in per_mag.items() if v < N_PER_OP_MAG]
        if not need:
            break
        try:
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if len(x) < sr:
                continue
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(p))[0][:20]
        peak = float(np.max(np.abs(x)))
        for key in rng.sample(need, min(len(need), 2)):
            op, mag_s = key.split("@")
            mag = float(mag_s)
            instr = next(i for m, i in MAGNITUDES[op] if m == mag)
            if op == "volume_up":
                if peak > 0:
                    limit = 0.94 / peak
                    want = 10 ** (mag / 20)
                    if want > limit:
                        continue  # нет запаса для этой величины
                out = (x * (10 ** (mag / 20))).astype(np.float32)
            elif op == "volume_down":
                out = (x * (10 ** (mag / 20))).astype(np.float32)
            elif op.startswith("pitch"):
                out = librosa.effects.pitch_shift(x, sr=sr, n_steps=mag).astype(np.float32)
            else:
                out = librosa.effects.time_stretch(x, rate=mag).astype(np.float32)
            tgt = os.path.join(OUT, f"{op}_{str(mag).replace('-','m').replace('.','p')}_{base}.wav")
            sf.write(tgt, out, sr)
            rows.append({"duration": round(len(out) / sr, 3), "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": instr},
                    {"type": "audio", "audio": p},
                ]},
                {"role": "assistant", "content": [{"type": "audio", "audio_url": tgt}]},
            ], "tool": op.split("_")[0], "meta": {
                "synthetic": True, "op": op, "params": {op.split("_")[1]: mag},
                "source_clip_id": base, "seed": 11}})
            per_mag[key] += 1
        if sum(per_mag.values()) % 1000 == 0 and sum(per_mag.values()) > 0:
            print(f"rows={sum(per_mag.values())}", flush=True)

    rng.shuffle(rows)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"rows": len(rows), "per_magnitude": per_mag,
              "hours": round(sum(r["duration"] for r in rows) / 3600, 2)}
    json.dump(report, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
