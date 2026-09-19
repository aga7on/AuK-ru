"""Best-of-N picking with GigaAM recall + pause profile (CPU, replaces Qwen ASR)."""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, r"G:\AI\AuK\local_train")
from gigaam_asr import transcribe  # noqa: E402

_NUM_WORDS = {
    "ноль", "один", "одна", "одно", "два", "две", "три", "четыре", "пять", "шесть", "семь",
    "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать", "двадцать",
    "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
    "девятьсот", "тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов",
}


def norm_words(text):
    import re
    text = text.lower().replace("ё", "е").replace("+", "").replace("'", "").replace("-", "")
    text = re.sub(r"[^а-я0-9\s]", " ", text)
    toks = []
    for w in text.split():
        if re.fullmatch(r"[0-9]+", w) or w in _NUM_WORDS:
            toks.append("<NUM>")
        else:
            toks.append(w)
    out = []
    for t in toks:
        if t == "<NUM>" and out and out[-1] == "<NUM>":
            continue
        out.append(t)
    return out


def recall_missing(ref_text, hyp_text):
    from collections import Counter
    rw, hw = norm_words(ref_text), norm_words(hyp_text)
    rc, hc = Counter(rw), Counter(hw)
    hits = sum((rc & hc).values())
    return hits / max(len(rw), 1), sorted((rc - hc).elements())


def silence_profile(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    frame, hop = 600, 240
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    hop_s = hop / sr
    lead = 0.0
    for i, s in enumerate(speech):
        if s:
            lead = i * hop_s
            break
    tail = 0.0
    for i in range(len(speech) - 1, -1, -1):
        if speech[i]:
            tail = (len(speech) - 1 - i) * hop_s
            break
    excess = max(0.0, lead - 0.3) + max(0.0, tail - 0.3)
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            d = (i - start) * hop_s
            p = start * hop_s
            if d >= 0.15 and p > lead + 0.2 and p < len(x) / sr - tail - 0.2:
                excess += max(0.0, d - 0.45)
            start = None
    return round(lead, 2), round(tail, 2), round(excess, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.dir, "manifest.json"), encoding="utf-8"))
    rows = []
    for m in man:
        hyp = transcribe(m["gen"])
        rec, miss = recall_missing(m["text"], hyp)
        lead, tail, excess = silence_profile(m["gen"])
        rows.append({"idx": m["idx"], "text": m["text"], "seed": m["seed"], "gen": m["gen"],
                     "asr": hyp, "recall": round(rec, 3), "missing": miss,
                     "lead": lead, "tail": tail, "excess": excess})
        print(f"c{m['idx']:02d} s{m['seed']}: recall={rec:.2f} excess={excess:.2f} missing={miss[:3]}")
        sys.stdout.flush()

    os.makedirs(os.path.join(args.dir, "best_gigaam"), exist_ok=True)
    picked = []
    for idx in sorted({r["idx"] for r in rows}):
        cand = [r for r in rows if r["idx"] == idx]
        best = sorted(cand, key=lambda r: (-r["recall"], r["excess"]))[0]
        dst = os.path.join(args.dir, "best_gigaam", f"best{idx:02d}.wav")
        shutil.copy(best["gen"], dst)
        picked.append(dict(best, best_file=dst))
        others = [f"s{c['seed']}:r{c['recall']}/e{c['excess']}" for c in cand if c is not best]
        print(f"PICK c{idx:02d}: seed={best['seed']} recall={best['recall']} excess={best['excess']} | {others}")
    json.dump(rows, open(os.path.join(args.dir, "scoring_gigaam.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(picked, open(os.path.join(args.dir, "picked_gigaam.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("GIGAAM_PICK_DONE")


if __name__ == "__main__":
    main()
