"""Инвентаризация RESD_Annotated: схема, эмоции, спикеры, длительности.
Достаём 3 примера wav в tmp для прослушивания. Никаких обучающих файлов не создаём."""
import io
import json
import os
from collections import Counter

import pyarrow.parquet as pq
import soundfile as sf

D = r"G:\AI\_datasets\aniemore_resd\data"
OUT = r"G:\AI\_AuK_tmp" if False else r"G:\AI\AuK\local_tests\resd_probe"

for f in ("train-00000-of-00001-1f5fe73d1293189c.parquet",):
    t = pq.read_table(os.path.join(D, f))
    print("schema:", t.schema.names, "rows:", t.num_rows)
    df = t.to_pandas()
    print(df.columns.tolist())
    print(df.head(3).to_dict("records") if len(df.columns) < 6 else [dict((c, str(df[c].iloc[i])[:80]) for c in df.columns) for i in range(3)])
    for col in df.columns:
        if df[col].dtype == object and df[col].map(type).eq(str).all() and df[col].nunique() < 30:
            print(col, dict(Counter(df[col])))
    # durations
    if "duration" in df.columns:
        print("dur:", df["duration"].describe().to_dict())
    # save 3 sample wavs
    os.makedirs(OUT, exist_ok=True)
    audio_col = next((c for c in df.columns if df[c].iloc[0] is not None and "audio" in c.lower() or c in ("bytes", "array")), None)
    print("audio col guess:", audio_col)
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        for c in df.columns:
            v = row[c]
            if isinstance(v, dict) and "bytes" in v:
                b = v["bytes"]
                p = os.path.join(OUT, f"sample_{i}.wav")
                open(p, "wb").write(b)
                try:
                    info = sf.info(p)
                    print(p, info.duration, "s", info.samplerate)
                except Exception as e:
                    print(p, "unreadable", e)
                break
