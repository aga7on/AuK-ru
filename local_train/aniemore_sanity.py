"""Sanity-check Aniemore на ИСТИННО эмоциональных клипах langswap (метка из датасета).
Ожидаем высокий hit-rate на teacher-данных, чтобы доверять гейту на генерациях.
"""
import json
import os
import random

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix")

rows = []
for l in open(os.path.join(SRC, "train.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r.get("meta", {}).get("source") == "langswap_dialogs" and r["meta"]["emotion"] in (
            "happy", "sad", "angry", "fear", "disgust", "surprise"):
        tgt = r["messages"][1]["content"][0]["audio_url"]
        rows.append({"emotion": r["meta"]["emotion"], "file": tgt})
random.Random(7).shuffle(rows)
rows = rows[:100]
json.dump(rows, open(os.path.join(AUK, "local_tests", "tmp_emotion_gate_check.json"), "w", encoding="utf-8"))
print("rows:", len(rows))
