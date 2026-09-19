"""Полная проверка пересечений: tools v3 × speech (train/val/dev/final) + линейность volume."""
import json
import os

import numpy as np

LOCAL = r"G:\AI\AuK\local_train"
TOOLS = os.path.join(LOCAL, "data_s2_tools_v3")
SPEECH = os.path.join(LOCAL, "data_s2_full")

# все аудио-пути speech
speech_paths = set()
for fn in ("train.jsonl", "val.jsonl"):
    with open(os.path.join(SPEECH, fn), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for msg in o["messages"]:
                for c in msg["content"]:
                    if c.get("type") == "audio":
                        speech_paths.add(c.get("audio") or c.get("audio_url"))
print(f"speech paths: {len(speech_paths)}")

# dev/final из split_manifest
dev_fin_paths = set()
with open(os.path.join(LOCAL, "corpus", "split_manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["split"] in ("dev", "final"):
            dev_fin_paths.add(o["p"])
print(f"dev/final paths: {len(dev_fin_paths)}")

# tools: все входы и цели
tools_all = set()
tools_rows = 0
for fn in ("train.jsonl", "val.jsonl"):
    with open(os.path.join(TOOLS, fn), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            tools_rows += 1
            for msg in o["messages"]:
                for c in msg["content"]:
                    if c.get("type") == "audio":
                        tools_all.add(c.get("audio") or c.get("audio_url"))
print(f"tools rows: {tools_rows}, paths: {len(tools_all)}")

# пересечения
overlap_speech = tools_all & speech_paths
overlap_devfin = tools_all & dev_fin_paths
print(f"tools × speech(train/val): {len(overlap_speech)}")
print(f"tools × dev/final: {len(overlap_devfin)}")
if overlap_speech:
    print("  examples:", list(overlap_speech)[:3])

# volume_up линейность (все файлы)
import soundfile as sf
vol_files = []
with open(os.path.join(TOOLS, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["meta"]["op"] == "volume_up":
            src = o["messages"][0]["content"][1]["audio"]
            tgt = o["messages"][1]["content"][0]["audio_url"]
            vol_files.append((src, tgt))

bad_linear = 0
bad_gain = 0
for src, tgt in vol_files[:200]:
    x, sr = sf.read(src, dtype="float32")
    y, _ = sf.read(tgt, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = min(len(x), len(y))
    gain = y[:n] / np.maximum(np.abs(x[:n]), 1e-9)
    med_gain = float(np.median(gain))
    db = 20 * np.log10(med_gain) if med_gain > 0 else -99
    if abs(db - 6.0) > 0.06:
        bad_gain += 1
    # линейность: y ≈ x * med_gain
    err = float(np.max(np.abs(y[:n] - x[:n] * med_gain)))
    if err > 0.01:
        bad_linear += 1

print(f"\nvolume_up checked: {min(200, len(vol_files))}")
print(f"gain != +6dB: {bad_gain}")
print(f"nonlinear: {bad_linear}")
