# -*- coding: utf-8 -*-
"""S16c step 2: ASR-верификация read-кандидатов (только читаная речь Common Voice).

Балансировка отбора по корзинам: на корзину cap --per_basket (по умолчанию 900),
чтобы всего верифицировать ~5-6k клипов (ASR стоит ~0.3 c/клип).
Фильтр: GigaAM WER ≤ wer_max И нет плохих подмен (ы→и / ч-ц / ш-щ / ж-з).

usage: python s16c_verify.py [--per_basket 900]
Выход: local_train/data_s2_full/v16c_read/read_clean.jsonl
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
CAND = os.path.join(AUK, "local_train", "data_s2_full", "v16c_read", "read_candidates.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16c_read", "read_clean.jsonl")

BAD_SUBS = [(r"[ыи]", r"[ыи]"), (r"[чц]", r"[чц]"), (r"[шщ]", r"[шщ]"), (r"[жз]", r"[жз]")]
TOP_WORDS = {"сегодня", "университета", "тебя", "холодильник", "кажется", "сколько",
             "работает", "целый", "лучше", "уже", "тысяча", "командировку", "пять",
             "автоматически", "встреча", "рублей"}


def norm(w):
    return re.sub(r"[^а-яё]", "", w.lower().replace("ё", "е"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_basket", type=int, default=900)
    ap.add_argument("--wer_max", type=float, default=0.10)
    args = ap.parse_args()

    from ru_metrics import transcribe_path, text_metrics

    rows = [json.loads(l) for l in open(CAND, encoding="utf-8") if l.strip()]
    print("read candidates:", len(rows))

    # балансировка: приоритет редким корзинам (yi/stress/topwords/devoicing),
    # затем клипам с топ-словами и длинными словами
    by_b = defaultdict(list)
    for r in rows:
        for b in r["baskets"]:
            by_b[b].append(r)
    rare_first = ["yi", "stress", "topwords", "devoicing", "clusters", "long_num", "softness"]

    def score(r):
        ws = re.findall(r"[а-яё]+", r["text"].lower())
        return -(len(set(ws) & TOP_WORDS) * 3 + sum(1 for w in ws if len(w) >= 9))

    picked = {}
    for b in rare_first:
        rs = sorted(by_b.get(b, []), key=score)
        for r in rs[: args.per_basket]:
            picked[r["path"]] = r
    todo = list(picked.values())
    print(f"to verify (balanced): {len(todo)}")

    clean = []
    n_err = 0
    wer_hist = Counter()
    for i, r in enumerate(todo):
        if not os.path.exists(r["path"]):
            n_err += 1
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
                          "wer": round(tm["wer"], 3), "baskets": r["baskets"]})
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(todo)}] clean={len(clean)} err={n_err}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    bc = Counter()
    for c in clean:
        bc.update(c["baskets"])
    print(f"S16C VERIFY DONE clean={len(clean)} ({sum(c['dur'] for c in clean)/3600:.2f}h) errors={n_err}")
    print("clean per basket:", dict(bc))
    print("wer hist:", dict(sorted(wer_hist.items())))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
