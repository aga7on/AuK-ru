"""Точная проверка volume_up: peak-ratio и least-squares gain."""
import json
import os

import numpy as np
import soundfile as sf

TOOLS = r"G:\AI\AuK\local_train\data_s2_tools_v3"
rows = []
with open(os.path.join(TOOLS, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["meta"]["op"] == "volume_up":
            rows.append(o)
        if len(rows) >= 8:
            break

print(f"volume_up rows: {len(rows)}")
for o in rows:
    src = o["messages"][0]["content"][1]["audio"]
    tgt = o["messages"][1]["content"][0]["audio_url"]
    x, sr = sf.read(src, dtype="float32")
    y, _ = sf.read(tgt, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)
    xp = float(np.max(np.abs(x)))
    yp = float(np.max(np.abs(y)))
    peak_db = 20 * np.log10(yp / max(xp, 1e-9))
    n = min(len(x), len(y))
    g = float(np.dot(y[:n], x[:n]) / max(np.dot(x[:n], x[:n]), 1e-12))
    ls_db = 20 * np.log10(abs(g)) if g != 0 else -99
    meta_gain = o["meta"].get("verified_gain_db")
    name = os.path.basename(tgt)[:38]
    print(f"{name}: src={xp:.3f} tgt={yp:.3f} peak_db={peak_db:+.3f} ls_db={ls_db:+.3f} meta={meta_gain}")
