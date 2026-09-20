# -*- coding: utf-8 -*-
"""S16b-микс (v16b_diction_mix): СБАЛАНСИРОВАННЫЙ фонетический буткемп по шести корзинам.

Источники:
  - baskets.jsonl     — 3044 клипа, классифицированы по корзинам (yi/softness/clusters/
                        devoicing/stress/long_num/topwords);
  - deficit_clean.jsonl — 1137 клипов, добытых прицельно на дефицитные корзины
                        (stress/devoicing) из полного корпуса, ASR-верифицированы.

Балансировка: целевое число клипов на корзину TARGET_PER_BASKET; repeat = clamp(
  ceil(target/count), 1, MAX_REPEAT). Клип может входить в несколько корзин —
  его repeat = максимум по его корзинам.

Защиты:
  - frozen hard_eval_v1: исключение по 4-граммам (ПРОВЕРЕНО: 0 пересечений);
  - тексты уже в v7 — не дублируются;
  - опциональный --filter-profanity (по умолчанию ВЫКЛ — решение пользователя 20.09).

База: v7_s7_mix 1:1 (сохраняет эмоции/tools/clone v1.0).
Выход: local_train/data_s2_full/v16b_diction_mix/train.jsonl + BASKETS_REPORT.md
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"G:\AI\AuK\src")

AUK = r"G:\AI\AuK"
DICT = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction")
SRC_V7 = os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl")
FROZEN = os.path.join(AUK, "local_train", "hard_eval_v1", "hard_eval_v1.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16b_diction_mix")
BASKETS = ["yi", "softness", "clusters", "devoicing", "stress", "long_num", "topwords"]
INSTR = "Say the following in Russian with clear, natural pronunciation: '{}'"

BAD_RE = re.compile(
    r"(ху[йяюеи]|бл[яa][дt]|пизд|еба|ёба|заеб|наху|сук[аи]|муда|гандон|пид[оa]р|"
    r"дерьм|жоп[ауы]|член|секс|траха|шлюх|херн|поху|охуе|долбоеб|ебать)", re.I)


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
    ap.add_argument("--filter-profanity", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    # --- frozen guard ---
    frozen_4g = set()
    for l in open(FROZEN, encoding="utf-8"):
        frozen_4g |= fourgrams(norm(json.loads(l)["text"]))
    print("frozen 4-grams:", len(frozen_4g))

    # --- v7 base + existing audio/texts ---
    v7 = [json.loads(l) for l in open(SRC_V7, encoding="utf-8") if l.strip()]
    v7_audio, v7_texts = set(), set()
    for r in v7:
        for m in r["messages"]:
            for c in m["content"]:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "audio":
                    u = c.get("audio") or c.get("audio_url")
                    if u:
                        v7_audio.add(os.path.normcase(u))
                elif c.get("type") == "text":
                    mm = re.search(r"'([^']*)'", c["text"])
                    if mm:
                        v7_texts.add(norm(mm.group(1)))
    print("v7 rows:", len(v7))

    # --- pool clips from both sources ---
    pool = {}   # path -> record
    for l in open(os.path.join(DICT, "baskets.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        pool[r["path"]] = {"path": r["path"], "text": r["text"], "dur": r["dur"],
                           "baskets": [b for b in r["baskets"] if b in BASKETS]}
    # все батчи дефицитной верификации (deficit_clean.jsonl + deficit_clean_batch*.jsonl)
    import glob
    deficit_files = [os.path.join(DICT, "deficit_clean.jsonl")] + \
                    sorted(glob.glob(os.path.join(DICT, "deficit_clean_batch*.jsonl")))
    seen_d = set()
    n_def = 0
    for df in deficit_files:
        if not os.path.exists(df) or df in seen_d:
            continue
        seen_d.add(df)
        for l in open(df, encoding="utf-8"):
            r = json.loads(l)
            cur = pool.get(r["path"])
            bs = sorted(set((cur["baskets"] if cur else []) + [b for b in r["baskets"] if b in BASKETS]))
            pool[r["path"]] = {"path": r["path"], "text": r["text"], "dur": r["dur"], "baskets": bs}
            n_def += 1
    print("deficit files merged:", len(seen_d), "records:", n_def)

    # filters
    keep = []
    dropped = Counter()
    for r in pool.values():
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
        if args.filter_profanity and BAD_RE.search(r["text"] or ""):
            dropped["profanity"] += 1
            continue
        if not r["baskets"]:
            dropped["no_basket"] += 1
            continue
        keep.append(r)
    print("pool:", len(pool), "| keep:", len(keep), "| dropped:", dict(dropped))

    # --- balance: repeat per basket ---
    counts = Counter()
    for r in keep:
        for b in r["baskets"]:
            counts[b] += 1
    rep_basket = {}
    for b in BASKETS:
        c = counts.get(b, 0)
        rep_basket[b] = 1 if c == 0 else max(1, min(args.max_repeat,
                                                    math.ceil(args.target_per_basket / c)))
    print("basket counts:", {b: counts.get(b, 0) for b in BASKETS})
    print("basket repeats:", rep_basket)

    rows = []
    per_basket_rows = Counter()
    for r in keep:
        rep = max(rep_basket[b] for b in r["baskets"])
        text = accentize(r["text"].strip())
        row = {"duration": float(r["dur"]),
               "messages": [
                   {"role": "user", "content": [{"type": "text", "text": INSTR.format(text)}]},
                   {"role": "assistant", "content": [{"type": "audio", "audio_url": r["path"]}]}],
               "split": "train",
               "meta": {"source": "s16b_phonetic_bootcamp", "baskets": r["baskets"]}}
        rows += [row] * rep
        for b in r["baskets"]:
            per_basket_rows[b] += rep

    out = v7 + rows
    import random
    random.Random(args.seed).shuffle(out)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours_add = sum(float(r["duration"]) for r in rows) / 3600
    hours_all = sum(float(r.get("duration") or 0) for r in out) / 3600
    L = ["# S16b MIX — сбалансированный фонетический буткемп", "",
         f"База v7: {len(v7)} строк (1:1, сохранены эмоции/tools/clone v1.0)",
         f"Добавка: {len(rows)} строк ({len(keep)} уникальных клипов, {hours_add:.1f} ч)",
         f"ИТОГО: {len(out)} строк, {hours_all:.1f} ч, доля добавки {len(rows)/len(out):.3f}", "",
         "## Корзины (клипов / строк после балансировки)", "",
         "| корзина | клипов | repeat | строк | часов |", "|---|---|---|---|---|"]
    for b in BASKETS:
        c = counts.get(b, 0)
        n = per_basket_rows.get(b, 0)
        hrs = sum(float(r["dur"]) * max(rep_basket[x] for x in r["baskets"])
                  for r in keep if b in r["baskets"]) / 3600
        L.append(f"| {b} | {c} | ×{rep_basket[b]} | {n} | {hrs:.2f} |")
    L += ["", "## Защиты", "",
          f"- frozen hard_eval_v1 (4-граммы): исключено {dropped.get('frozen_4gram', 0)}",
          f"- уже в v7 (аудио): {dropped.get('already_in_v7_audio', 0)}; (тексты): {dropped.get('text_already_in_v7', 0)}",
          f"- отсутствуют файлы: {dropped.get('missing_file', 0)}",
          f"- фильтр нецензурного: {'ВКЛ' if args.filter_profanity else 'ВЫКЛ (решение пользователя 20.09)'}",
          f"  (отброшено {dropped.get('profanity', 0)})", "",
          "Целевой объект — фонетический КОНТРАСТ/семейство, а не конкретное слово или строка.",
          "Гейт: hard_eval_v1_clean (235) + human blind test; стоп-условие TTS first_ok ≥ 0.812."]
    open(os.path.join(OUT, "BASKETS_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    print("out rows:", len(out), "| added:", len(rows), f"| add hours {hours_add:.1f} | total {hours_all:.1f}")
    print("basket rows:", dict(per_basket_rows))
    print("wrote", os.path.join(OUT, "train.jsonl"))


if __name__ == "__main__":
    main()
