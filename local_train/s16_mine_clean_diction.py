# -*- coding: utf-8 -*-
"""S16 step 1: майнинг аудио с ПРОВЕРЕННО чистой дикцией для фонетического буткемпа.

Идея (отличие от S8, который отбирал по трудности ТЕКСТА и сломал first_ok):
отбираем по качеству ПРОИЗНОШЕНИЯ в аудио. Клип попадает в набор, если GigaAM
распознал его с WER ≤ 0.10 И без «плохих» фонемных подмен (ы→и, ч/ц, ш/щ, оглушение).
Дополнительно помечаем клипы, содержащие трудные слова (топ words_mangled всех прогонов) —
это золотые примеры «трудное слово, произнесённое чисто».

usage: python s16_mine_clean_diction.py [--limit 4000] [--out ...]
Выход: local_train/data_s2_full/v16_diction/clean_clips.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
SELECTED = os.path.join(AUK, "local_train", "data", "selected.jsonl")
OUT_DIR = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction")

# Топ системно трудных слов (PRONUNCIATION_AXES.md) + фонемно-опасные паттерны
HARD_WORDS = {"сегодня", "университета", "университет", "тебя", "холодильник", "кажется",
              "сколько", "работает", "целый", "лучше", "уже", "тысяча", "командировку",
              "предыдущего", "зарегистрировались", "ключи", "мы", "пять", "учится",
              "автоматически", "встреча", "вторая", "рублей", "петровна"}

# «плохие» подмены, которые мы НЕ хотим закреплять (паттерны пользователя + судьи)
BAD_SUBS = [
    (r"[ыи]", r"[ыи]"),   # ы<->и («мы»->«ми», «свои»->«сои»)
    (r"[чц]", r"[чц]"),   # ч<->ц («ключи»->«клучи»)
    (r"[шщ]", r"[шщ]"),
    (r"[жз]", r"[жз]"),
]


def norm(w):
    return re.sub(r"[^а-яё]", "", w.lower().replace("ё", "е"))


def bad_substitutions(ref_text, heard):
    """Возвращает число подмен, совпавших с BAD_SUBS-паттернами."""
    from ru_metrics import text_metrics
    tm = text_metrics(ref_text, heard)
    bad = 0
    for r, h in tm.get("substitutions", []):
        rn, hn = norm(r), norm(h)
        if not rn or not hn or rn == hn:
            continue
        for rp, hp in BAD_SUBS:
            if re.fullmatch(rp, rn) and re.fullmatch(hp, hn) and rn != hn:
                bad += 1
                break
    return bad, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000, help="макс. клипов для транскрибации")
    ap.add_argument("--wer_max", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "clean_clips.jsonl"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    from ru_metrics import transcribe_path

    rows = [json.loads(l) for l in open(SELECTED, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ok") and (r.get("ovrl") or 0) >= 2.8
            and 2.0 <= (r.get("dur") or 0) <= 15.0]
    # приоритет: клипы с трудными словами (их транскрибируем первыми)
    def hard_score(r):
        ws = set(norm(w) for w in re.findall(r"[а-яё]+", (r.get("text") or "").lower()))
        return -len(ws & HARD_WORDS)
    rows.sort(key=hard_score)
    rows = rows[: args.limit]
    print(f"candidates to transcribe: {len(rows)} (limit {args.limit})")

    clean = []
    hard_clean = 0
    n_err = 0
    wer_hist = Counter()
    for i, r in enumerate(rows):
        if not os.path.exists(r["path"]):
            continue
        heard, err = transcribe_path(r["path"])
        if err:
            n_err += 1
            continue
        bad, tm = bad_substitutions(r["text"], heard)
        wer = tm["wer"]
        wer_hist[min(int(wer * 10), 10)] += 1
        if wer <= args.wer_max and bad == 0:
            ws = set(norm(w) for w in re.findall(r"[а-яё]+", (r.get("text") or "").lower()))
            has_hard = bool(ws & HARD_WORDS)
            hard_clean += has_hard
            clean.append({"path": r["path"], "text": r["text"], "dur": round(r["dur"], 2),
                          "ovrl": r.get("ovrl"), "gender": r.get("gender"),
                          "wer": wer, "hard_words": sorted(ws & HARD_WORDS)})
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(rows)}] clean={len(clean)} hard_clean={hard_clean} err={n_err}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    hours = sum(c["dur"] for c in clean) / 3600
    print(f"DONE transcribed={len(rows)} clean={len(clean)} ({hours:.1f}h) hard_clean={hard_clean} errors={n_err}")
    print("wer histogram (deciles):", dict(sorted(wer_hist.items())))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
