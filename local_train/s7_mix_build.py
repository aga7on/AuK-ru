"""s7-микс: v5_s5_mix (s4-микс + replay + клон-буст) + ЭМОЦИОНАЛЬНЫЕ пары из langswap dialogs.

Формат эмо-пары (как учит upstream): user = англ. инструкция "Say in Russian with a {emo} tone: '<accent_text>'"
+ референс-аудио ТОГО ЖЕ спикера (нейтральный клип), assistant = эмо-клик 24кГц.
Рефы: другой клип того же speaker_id (предпочтительно neutral), обрезка до 20с.
Neutral-клипы: клон-формат без эмо-слова (обычная инструкция).
Дублирование эмо-классов ×N для баланса против 37к базовых строк.

Выход: local_train/data_s2_full/v7_s7_mix/train.jsonl
"""
import json
import os
import random
from collections import Counter, defaultdict

import soundfile as sf

AUK = r"G:\AI\AuK"
SRC = r"G:\AI\_datasets\dialogs_emotional"
BASE = os.path.join(AUK, "local_train", "data_s2_full", "v5_s5_mix", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix")
SEED = 7

# повтор эмо-классов: редкие усиливаем, чтобы получить суммарно ~8-10% микса
REPEATS = {"happy": 6, "sad": 8, "angry": 10, "fear": 12, "disgust": 10,
           "surprise": 6, "whisper": 14, "laughing": 10, "neutral_emo": 1}
MAX_REF_S = 20.0
MAX_TGT_S = 18.0

EMO_EN = {"happy": "happy", "sad": "sad", "angry": "angry", "fear": "fearful",
          "disgust": "disgusted", "surprise": "surprised", "whisper": "whispering",
          "laughing": "laughing", "neutral": "neutral"}


def load_rows():
    lines = open(os.path.join(SRC, "downloaded_rows.csv"), encoding="utf-8").read().splitlines()
    hdr = lines[0].split("|")
    rows = []
    for l in lines[1:]:
        p = l.split("|")
        if len(p) == 7:
            rows.append(dict(zip(hdr, p)))
    return rows


def find_audio(rel):
    p = os.path.join(SRC, rel)
    return p if os.path.exists(p) and os.path.getsize(p) > 1000 else None


def trim_wav(src, dst, max_s):
    if os.path.exists(dst):
        return dst
    a, sr = sf.read(src)
    if len(a) / sr > max_s:
        a = a[: int(max_s * sr)]
    sf.write(dst, a, sr)
    return dst


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "_refs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "_tgt"), exist_ok=True)
    rng = random.Random(SEED)

    base = [json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()]
    rows = load_rows()
    print("base", len(base), "| emo rows", len(rows), flush=True)

    # группируем по спикеру; отдельно neutral-пул для рефов
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    neutral_pool = {r["audio_path"] for r in rows if r["emotion"] == "neutral"}

    emo_rows = []
    used_refs = 0
    for spk, items in by_spk.items():
        # рефы: нейтральные клипы этого спикера, иначе любые другие его клипы
        ref_candidates = [r for r in items if r["emotion"] == "neutral" and find_audio(r["audio_path"])]
        if not ref_candidates:
            ref_candidates = [r for r in items if find_audio(r["audio_path"])]
        if not ref_candidates:
            continue
        rng.shuffle(ref_candidates)
        for it in items:
            tgt = find_audio(it["audio_path"])
            if not tgt:
                continue
            # длительность цели
            try:
                d = sf.info(tgt).duration
            except Exception:
                continue
            if not (1.5 <= d <= MAX_TGT_S):
                continue
            ref = ref_candidates[rng.randrange(len(ref_candidates))]
            if ref["audio_path"] == it["audio_path"]:
                continue
            ref_file = find_audio(ref["audio_path"])
            if not ref_file:
                continue
            stem = os.path.splitext(os.path.basename(it["audio_path"]))[0]
            ref_dst = os.path.join(OUT, "_refs", stem + "_ref.wav")
            tgt_dst = os.path.join(OUT, "_tgt", stem + "_tgt.wav")
            trim_wav(ref_file, ref_dst, MAX_REF_S)
            trim_wav(tgt, tgt_dst, MAX_TGT_S)
            used_refs += 1
            emo_rows.append({
                "duration": float(sf.info(tgt_dst).duration),
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"Reproduce the reference voice and say in Russian with a "
                            f"{EMO_EN.get(it['emotion'], it['emotion'])} tone: '{it['accent_text']}'")},
                        {"type": "audio", "audio": ref_dst},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "audio", "audio_url": tgt_dst},
                    ]},
                ],
                "split": "train",
                "meta": {"emotion": it["emotion"], "speaker": it["speaker_id"],
                         "source": "langswap_dialogs"},
            })

    by_emo = Counter(r["meta"]["emotion"] for r in emo_rows)
    print("emo pairs:", len(emo_rows), dict(by_emo), flush=True)

    # повторение с учётом квот
    repeated = []
    for r in emo_rows:
        rep = REPEATS.get(r["meta"]["emotion"], 4)
        repeated.extend([r] * rep)
    print("after repeats:", len(repeated), flush=True)

    mix = base + repeated
    rng.shuffle(mix)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration", 0)) for r in mix) / 3600
    bad = sum(1 for r in mix if not isinstance(r.get("duration"), float))
    report = {
        "base_rows": len(base), "emo_unique": len(emo_rows), "emo_by_class": dict(by_emo),
        "emo_rows_after_repeats": len(repeated), "total": len(mix),
        "hours": round(hours, 2), "non_float_duration": bad,
        "emo_share": round(len(repeated) / len(mix), 4), "seed": SEED,
    }
    json.dump(report, open(os.path.join(OUT, "mix_report.json"), "w", encoding="utf-8"), indent=1)
    print("MIX_DONE", json.dumps(report))


if __name__ == "__main__":
    main()
