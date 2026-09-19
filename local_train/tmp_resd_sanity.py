"""Контроль: Aniemore WavLM на RESD (домен обучения классификатора) — истинная точность гейта.
Если тут высокая, а на langswap 17% — гейт валиден, но не переносится на диалоговый домен.
"""
import ast, json, os, random, sys

sys.path.insert(0, r"G:\AI\AuK\local_train\thirdparty\aniemore_pkg")

import pyarrow.parquet as pq

AUK = r"G:\AI\AuK"
SRC = r"G:\AI\_datasets\aniemore_resd\data\train-00000-of-00001-1f5fe73d1293189c.parquet"
OUTW = os.path.join(AUK, "local_tests", "tmp_resd_clips")
os.makedirs(OUTW, exist_ok=True)

t = pq.read_table(SRC)
df = t.to_pandas()
rows = []
idx = list(range(len(df)))
random.Random(11).shuffle(idx)
for i in idx[:150]:
    r = df.iloc[i]
    speech = r["speech"]
    if isinstance(speech, str):
        speech = ast.literal_eval(speech)
    b = speech["bytes"] if isinstance(speech, dict) else b"".join(speech) if isinstance(speech, list) else speech
    fn = os.path.join(OUTW, f"{i:05d}.wav")
    if not os.path.exists(fn):
        open(fn, "wb").write(b)
    rows.append({"emotion": r["emotion"], "file": fn})
json.dump(rows, open(os.path.join(AUK, "local_tests", "tmp_emotion_gate_check_resd.json"), "w", encoding="utf-8"))
print("rows:", len(rows), "labels:", sorted(set(x["emotion"] for x in rows)))
