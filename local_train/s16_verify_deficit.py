# -*- coding: utf-8 -*-
"""ASR-верификация дефицитных корзин (stress/devoicing): оставляем только клипы
с ПРОВЕРЕННО чистой дикцией (GigaAM WER ≤ wer_max, без плохих подмен ы→и / ч-ц / ш-щ).

Семплирование: --per_basket N (по умолчанию 1200 на корзину) — чтобы не транскрибировать
все 19k; приоритет клипам, содержащим топ-слова и длинные слова.

usage: python s16_verify_deficit.py [--per_basket 1200] [--device cuda:1]
Выход: local_train/data_s2_full/v16_diction/deficit_clean.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
CAND = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "deficit_candidates.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "deficit_clean.jsonl")

BAD_SUBS = [
    (r"[ыи]", r"[ыи]"),
    (r"[чц]", r"[чц]"),
    (r"[шщ]", r"[шщ]"),
    (r"[жз]", r"[жз]"),
]
TOP_WORDS = {"сегодня", "университета", "тебя", "холодильник", "кажется", "сколько",
             "работает", "целый", "лучше", "уже", "тысяча", "командировку", "пять",
             "автоматически", "встреча", "рублей"}


def norm(w):
    return re.sub(r"[^а-яё]", "", w.lower().replace("ё", "е"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_basket", type=int, default=1200)
    ap.add_argument("--wer_max", type=float, default=0.10)
    args = ap.parse_args()

    from ru_metrics import transcribe_path, text_metrics

    rows = [json.loads(l) for l in open(CAND, encoding="utf-8") if l.strip()]
    by_b = defaultdict(list)
    for r in rows:
        for b in r["deficit_baskets"]:
            by_b[b].append(r)

    rng = random.Random(16)
    picked = {}
    for b, rs in by_b.items():
        # приоритет: больше топ-слов и длинных слов
        def score(r):
            ws = re.findall(r"[а-яё]+", r["text"].lower())
            return -(len(set(ws) & TOP_WORDS) * 3 + sum(1 for w in ws if len(w) >= 9))
        rs.sort(key=score)
        picked[b] = rs[: args.per_basket]
    todo = {}
    for b, rs in picked.items():
        for r in rs:
            todo[r["path"]] = r
    todo = list(todo.values())
    print(f"to verify: {len(todo)} (stress {len(picked.get('stress',[]))}, devoicing {len(picked.get('devoicing',[]))})")

    clean = []
    n_err = 0
    wer_hist = Counter()
    for i, r in enumerate(todo):
        if not os.path.exists(r["path"]):
            continue
        heard, err = transcribe_path(r["path"])
        if err:
            n_err += 1
            continue
        tm = text_metrics(r["text"], heard)
        wer_hist[min(int(tm["wer"] * 10), 10)] += 1
        bad = 0
        for a, b in tm.get("substitutions", []):
            an, bn = norm(a), norm(b)
            if not an or not bn or an == bn:
                continue
            for rp, hp in BAD_SUBS:
                if re.fullmatch(rp, an) and re.fullmatch(hp, bn):
                    bad += 1
                    break
        if tm["wer"] <= args.wer_max and bad == 0:
            clean.append({"path": r["path"], "text": r["text"], "dur": r["dur"],
                          "wer": round(tm["wer"], 3),
                          "baskets": r["deficit_baskets"]})
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(todo)}] clean={len(clean)} err={n_err}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    bc = Counter()
    for c in clean:
        bc.update(c["baskets"])
    print(f"VERIFY DONE clean={len(clean)} ({sum(c['dur'] for c in clean)/3600:.2f}h) errors={n_err}")
    print("clean per basket:", dict(bc))
    print("wer hist:", dict(sorted(wer_hist.items())))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
