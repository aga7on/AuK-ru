"""Скачивание langswap/dialogs-ru-emotional-conversations (HF) для s7 эмоций.
Приоритет: эмоциональные классы (happy/sad/angry/fear/disgust/surprise/whisper/laughing) +
neutral для баланса. Ограничение по времени скачивания.
Выход: G:\\AI\\_datasets\\dialogs_emotional\\ (wavs + metadata.csv)
"""
import io
import json
import os
import time
import urllib.request
from collections import Counter

BASE = "https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations/resolve/main"
OUT = r"G:\AI\_datasets\dialogs_emotional"
META = os.path.join(OUT, "metadata.csv")
WAVS = os.path.join(OUT, "wavs")

# целевые классы и квоты (neutral нужен как «обычная» речь для смешивания)
QUOTA = {"happy": 400, "sad": 300, "angry": 200, "fear": 128, "disgust": 181,
         "surprise": 400, "whisper": 59, "laughing": 150, "neutral": 600}


def fetch(url, retries=3):
    for i in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def main():
    os.makedirs(WAVS, exist_ok=True)
    if not os.path.exists(META):
        data = fetch(f"{BASE}/metadata.csv")
        open(META, "wb").write(data)
    lines = open(META, encoding="utf-8").read().splitlines()
    hdr, rows = lines[0], lines[1:]
    parsed = []
    for l in rows:
        p = l.split("|")
        if len(p) == 7:
            parsed.append(p)
    print("rows", len(parsed))

    picked = {k: [] for k in QUOTA}
    for p in parsed:
        emo = p[3]
        if emo in picked and len(picked[emo]) < QUOTA[emo]:
            picked[emo].append(p)

    todo = [p for ps in picked.values() for p in ps]
    print("to download:", len(todo), Counter(p[3] for p in todo))

    ok, fail = 0, 0
    t0 = time.time()
    report = []
    for i, p in enumerate(todo):
        rel = p[0]
        dst = os.path.join(WAVS, os.path.basename(rel))
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            ok += 1
            report.append(p)
            continue
        try:
            data = fetch(f"{BASE}/{rel}")
            open(dst, "wb").write(data)
            ok += 1
            report.append(p)
        except Exception as e:
            fail += 1
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(todo)}] ok={ok} fail={fail} ({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"DONE ok={ok} fail={fail}")
    with open(os.path.join(OUT, "downloaded_rows.csv"), "w", encoding="utf-8") as f:
        f.write(hdr + "\n")
        for p in report:
            f.write("|".join(p) + "\n")
    json.dump({"ok": ok, "fail": fail, "quota": QUOTA},
              open(os.path.join(OUT, "download_report.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
