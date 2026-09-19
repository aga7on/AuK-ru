"""Финальная проверка tools v3: все пересечения + volume линейность по пикам (весь набор)."""
import json
import os

import numpy as np
import soundfile as sf

LOCAL = r"G:\AI\AuK\local_train"
TOOLS = os.path.join(LOCAL, "data_s2_tools_v3")
SPEECH = os.path.join(LOCAL, "data_s2_full")

speech_paths = set()
for fn in ("train.jsonl", "val.jsonl"):
    with open(os.path.join(SPEECH, fn), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for msg in o["messages"]:
                for c in msg["content"]:
                    if c.get("type") == "audio":
                        speech_paths.add(c.get("audio") or c.get("audio_url"))

dev_fin_paths = set()
with open(os.path.join(LOCAL, "corpus", "split_manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["split"] in ("dev", "final"):
            dev_fin_paths.add(o["p"])

tools_all = set()
vol_up = []
for fn in ("train.jsonl", "val.jsonl"):
    with open(os.path.join(TOOLS, fn), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for msg in o["messages"]:
                for c in msg["content"]:
                    if c.get("type") == "audio":
                        tools_all.add(c.get("audio") or c.get("audio_url"))
            if o["meta"]["op"] == "volume_up":
                vol_up.append((o["messages"][0]["content"][1]["audio"],
                               o["messages"][1]["content"][0]["audio_url"]))

print(f"tools paths: {len(tools_all)}")
print(f"tools x speech(train/val): {len(tools_all & speech_paths)}")
print(f"tools x dev/final: {len(tools_all & dev_fin_paths)}")

# volume_up: peak-ratio по ВСЕМ файлам
bad = 0
for src, tgt in vol_up:
    x, _ = sf.read(src, dtype="float32")
    y, _ = sf.read(tgt, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)
    xp = float(np.max(np.abs(x)))
    yp = float(np.max(np.abs(y)))
    if xp < 1e-6:
        continue
    db = 20 * np.log10(yp / xp)
    if abs(db - 6.0) > 0.06:
        bad += 1
print(f"\nvolume_up files: {len(vol_up)}, gain!=+6dB: {bad}")

# клиппинг в volume_up
clip = sum(1 for _, tgt in vol_up[:500]
           if np.max(np.abs(sf.read(tgt, dtype="float32")[0])) > 0.9501)
print(f"volume_up peak>0.9501 (первый 500): {clip}")
