# -*- coding: utf-8 -*-
"""S16c-микс (v16c_read_mix): фонетический буткемп ТОЛЬКО на читаной речи (Common Voice).

Отличие от S16b (провалил гейт 3/10 из-за 67% спонтанной речи в датасете):
источник добавки — read_clean.jsonl (common_voice_ru_*, отфильтрованы филлеры/
самоисправления, ASR-верифицированы WER ≤0.10, без подмен ы→и/ч-ц/ш-щ).

База: v7_s7_mix 1:1 (эмоции/tools/clone v1.0 сохраняются).
Балансировка корзин: target_per_basket, repeat ≤ max_repeat.
Защиты: frozen hard_eval_v1 (4-граммы), дубликаты v7, отсутствующие файлы.

usage: python s16c_mix_build.py [--target_per_basket 2200] [--max_repeat 5]
Выход: local_train/data_s2_full/v16c_read_mix/train.jsonl + BASKETS_REPORT.md
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, r"G:\AI\AuK\src")

AUK = r"G:\AI\AuK"
SRC_V7 = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
READ_CLEAN = os.path.join(AUK, "local_train", "data_s2_full", "v16c_read", "read_clean.jsonl")
FROZEN = os.path.join(AUK, "local_train", "hard_eval_v1", "hard_eval_v1_clean.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16c_read_mix")
BASKETS = ["yi", "softness", "clusters", "devoicing", "stress", "long_num", "topwords"]
INSTR = "Say the following in Russian with clear, natural pronunciation: '{}'"


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "").replace("ё", "е")).strip()


def fourgrams(t):
    w = t.split()
    return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))} if len(w) >= 4 else set()


def accentize(text):
    try:
        from auk.infer.infer_gradio import _accentize_ru
        return _accentize_ru(text, use_lexicon=False)
    except Exception:
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_per_basket", type=int, default=2200)
    ap.add_argument("--max_repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=18)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    frozen_4g = set()
    for l in open(FROZEN, encoding="utf-8"):
        frozen_4g |= fourgrams(norm(json.loads(l)["text"]))
    print("frozen 4-grams:", len(frozen_4g))

    v7 = [json.loads(l) for l in open(SRC_V7, encoding="utf-8") if l.strip()]
    v7_audio, v7_texts = set(), set()
    for r in v7:
        for m in r["messages"]:
            for c in m["content"]:
                if isinstance(c, dict):
                    if c.get("type") == "audio":
                        u = c.get("audio") or c.get("audio_url")
                        if u:
                            v7_audio.add(os.path.normcase(u))
                    elif c.get("type") == "text":
                        mm = re.search(r"'([^']*)'", c["text"])
                        if mm:
                            v7_texts.add(norm(mm.group(1)))
    print("v7 rows:", len(v7))

    clean = [json.loads(l) for l in open(READ_CLEAN, encoding="utf-8") if l.strip()]
    print("read clean clips:", len(clean))

    keep, dropped = [], Counter()
    for r in clean:
        if os.path.normcase(r["path"]) in v7_audio:
            dropped["already_in_v7_audio"] += 1
            continue
        if not os.path.exists(r["path"]):
            dropped["missing_file"] += 1
            continue
        t = norm(r["text"])
        if fourgrams(t) & frozen_4g:
            dropped["frozen_4gram"] += 1
            continue
        if t in v7_texts:
            dropped["text_already_in_v7"] += 1
            continue
        bs = [b for b in (r.get("baskets") or []) if b in BASKETS]
        if not bs:
            dropped["no_basket"] += 1
            continue
        keep.append({**r, "baskets": bs})
    print("keep:", len(keep), "| dropped:", dict(dropped))

    counts = Counter()
    for r in keep:
        counts.update(r["baskets"])
    rep = {b: (1 if counts.get(b, 0) == 0 else
               max(1, min(args.max_repeat, math.ceil(args.target_per_basket / counts[b]))))
           for b in BASKETS}
    print("basket counts:", {b: counts.get(b, 0) for b in BASKETS})
    print("basket repeats:", rep)

    rows, per_b = [], Counter()
    for r in keep:
        k = max(rep[b] for b in r["baskets"])
        text = accentize(r["text"].strip())
        row = {"duration": float(r["dur"]),
               "messages": [
                   {"role": "user", "content": [{"type": "text", "text": INSTR.format(text)}]},
                   {"role": "assistant", "content": [{"type": "audio", "audio_url": r["path"]}]}],
               "split": "train",
               "meta": {"source": "s16c_read_speech", "baskets": r["baskets"], "wer": r.get("wer")}}
        rows += [row] * k
        for b in r["baskets"]:
            per_b[b] += k

    out = v7 + rows
    random.Random(args.seed).shuffle(out)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hrs_add = sum(float(r["duration"]) for r in rows) / 3600
    hrs_all = sum(float(r.get("duration") or 0) for r in out) / 3600
    L = ["# S16c MIX — фонетический буткемп на ЧИТАНОЙ речи (Common Voice)", "",
         f"База v7: {len(v7)} строк (1:1 — эмоции/tools/clone v1.0 сохранены)",
         f"Добавка: {len(rows)} строк ({len(keep)} уникальных клипов, {hrs_add:.1f} ч)",
         f"ИТОГО: {len(out)} строк, {hrs_all:.1f} ч, доля добавки {len(rows)/len(out):.3f}", "",
         "Источник добавки: `read_clean.jsonl` — ТОЛЬКО common_voice_ru_* (читаная речь),",
         "без филлеров/самоисправлений, ASR-верифицировано (WER ≤0.10, без подмен ы→и/ч-ц/ш-щ).",
         "Исправление ошибки S16b, где 67% добавки было спонтанной речью (S16_RESULTS.md).", "",
         "## Корзины", "", "| корзина | клипов | repeat | строк |", "|---|---|---|---|"]
    for b in BASKETS:
        L.append(f"| {b} | {counts.get(b,0)} | ×{rep[b]} | {per_b.get(b,0)} |")
    L += ["", "## Защиты", "",
          f"- frozen hard_eval_v1_clean (4-граммы): исключено {dropped.get('frozen_4gram',0)}",
          f"- уже в v7 (аудио/тексты): {dropped.get('already_in_v7_audio',0)} / {dropped.get('text_already_in_v7',0)}",
          f"- файлы отсутствуют: {dropped.get('missing_file',0)}", "",
          "Гейт S16c: CER hard ≤ 0.0703 (v1.0), naturalness/подмены/mangled не хуже baseline,",
          "TTS first_ok ≥ 0.812, TTS WER ≤ 0.077, эмо годен ≥ 48.3%.",
          "Правило остановки: если hard-CER не улучшится и на читаной речи — ветка «данные» закрыта."]
    open(os.path.join(OUT, "BASKETS_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    print("out rows:", len(out), "| added:", len(rows), f"| add {hrs_add:.1f}h | total {hrs_all:.1f}h")
    print("basket rows:", dict(per_b))
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
