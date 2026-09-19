import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")

from auk.infer.ru_translit import ru_to_latin

SRC = r"G:\AI\AuK\local_train\data\candidates.jsonl"
OUT_DIR = r"G:\AI\AuK\local_train\smoke"

TEMPLATES = [
    "Say the following in Russian with clear, natural pronunciation: '{t}'",
    "Speak the following Russian text aloud in a natural voice: '{t}'",
    "Pronounce the following in Russian: '{t}'",
    "Read the following Russian sentence out loud: '{t}'",
]

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")][:40]
os.makedirs(OUT_DIR, exist_ok=True)


def write(rows_slice, name, offset=0):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows_slice):
            latin = ru_to_latin(r["text"])
            instr = TEMPLATES[(i + offset) % len(TEMPLATES)].format(t=latin)
            f.write(json.dumps({
                "duration": r["dur"],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": instr}]},
                    {"role": "assistant", "content": [{"type": "audio", "audio_url": r["path"]}]},
                ],
            }, ensure_ascii=False) + "\n")


write(rows[4:], "train.jsonl", offset=0)
write(rows[:4], "val.jsonl", offset=1)
print("smoke data ready:", len(rows) - 4, "train /", 4, "val")
