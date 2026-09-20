# -*- coding: utf-8 -*-
"""S12-микс (v12_emo_mix): Emotion v2 — оси emotion × intensity × speaking_mode (S12_DESIGN.md).

Основа: v7_s7_mix (51134 строк). Преобразования emo-строк:
  1. intensity-прокси: RMS dB активных сегментов tgt-клипа; тертили ВНУТРИ (speaker, emotion)
     → low/mid/high. Формы инструкции:
       low  → "with a {emo} tone, subdued:"
       mid  → "with a {emo} tone:"          (совместимо с v1.0-инструкциями)
       high → "with a {emo} tone, intense:"
  2. speaking_mode: whisper/laughing-строки → отдельная ось:
       "say in Russian, whispering: '<text>'" / ", laughing:" (без слова tone — не эмоция).
  3. neutral_emo и остальные строки (tts/clone/tool) — без изменений.

Не-emo строки сохраняются 1:1 (emotion replay уже внутри; neutral/tool — как в v7).
Выход: local_train/data_s2_full/v12_emo_mix/train.jsonl
"""
import json
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v12_emo_mix")

_rms_cache = {}


def rms_db(path):
    if path in _rms_cache:
        return _rms_cache[path]
    try:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        act = x[np.abs(x) > 0.005]
        v = float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12)) if len(act) else -60.0
    except Exception:
        v = -60.0
    _rms_cache[path] = v
    return v


def get_text(r):
    for m in r["messages"]:
        if m["role"] == "user":
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "text":
                    return c["text"]
    return None


def set_text(r, t):
    for m in r["messages"]:
        if m["role"] == "user":
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "text":
                    c["text"] = t


def get_tgt(r):
    for m in r["messages"]:
        if m["role"] == "assistant":
            for c in m["content"]:
                if isinstance(c, dict):
                    return c.get("audio_url") or c.get("audio")
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print("v7 rows:", len(rows))

    # индекс emo-строк и их RMS
    emo_idx = []
    for i, r in enumerate(rows):
        meta = r.get("meta") or {}
        emo = meta.get("emotion")
        if not emo:
            continue
        emo_idx.append(i)
    print("emo rows:", len(emo_idx))

    # RMS по группам (speaker, emotion)
    groups = defaultdict(list)
    for i in emo_idx:
        r = rows[i]
        meta = r["meta"]
        tgt = get_tgt(r)
        groups[(meta.get("speaker"), meta.get("emotion"))].append((i, rms_db(tgt) if tgt else -60.0))

    # тертили → intensity
    label = {}
    for key, vals in groups.items():
        xs = sorted(v for _, v in vals)
        if len(xs) >= 3:
            t1 = xs[len(xs) // 3]
            t2 = xs[2 * len(xs) // 3]
        else:
            t1 = t2 = xs[0] if xs else 0
        for i, v in vals:
            label[i] = "low" if v <= t1 else ("high" if v >= t2 and v > t1 else "mid")

    from collections import Counter
    print("intensity dist:", dict(Counter(label.values())))

    # переписываем инструкции
    n_emo_rewrite = n_mode_rewrite = 0
    for i in emo_idx:
        r = rows[i]
        meta = r["meta"]
        emo = meta["emotion"]
        txt = get_text(r)
        if txt is None:
            continue
        m = re.search(r"with a (\w+) tone:", txt)
        if emo in ("whisper", "laughing"):
            mode = "whispering" if emo == "whisper" else "laughing"
            if m:
                # «and say in Russian with a whispering tone:» → «and say in Russian, whispering:»
                new = txt.replace(" " + m.group(0), f", {mode}:")
                set_text(r, new)
                r["meta"]["speaking_mode"] = mode
                r["meta"].pop("emotion", None)
                n_mode_rewrite += 1
            continue
        lv = label.get(i, "mid")
        if m:
            w = m.group(1)
            if lv == "low":
                new = txt.replace(f"with a {w} tone:", f"with a {w} tone, subdued:")
            elif lv == "high":
                new = txt.replace(f"with a {w} tone:", f"with a {w} tone, intense:")
            else:
                new = txt
            set_text(r, new)
            r["meta"]["intensity"] = lv
            n_emo_rewrite += 1
    print("emo intensity rewrites:", n_emo_rewrite, "| speaking_mode rewrites:", n_mode_rewrite)

    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration") or 0) for r in rows) / 3600
    print("rows:", len(rows), "hours: %.1f" % hours)
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
