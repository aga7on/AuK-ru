"""GigaAM recall table for autopsy files (objective reference for honest listening)."""
import json
import os
import re
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train")
from gigaam_asr import transcribe  # noqa: E402

D = r"G:\AI\AuK\local_tests\autopsy"
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
man = sorted(man, key=lambda m: (m["idx"], m["seed"]))

_NUM_WORDS = {
    "ноль", "один", "одна", "одно", "два", "две", "три", "четыре", "пять", "шесть", "семь",
    "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать", "двадцать",
    "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
    "девятьсот", "тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов",
}


def norm_words(text):
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


rows = []
for m in man:
    hyp = transcribe(m["gen"])
    rec, miss = recall_missing(m["text"], hyp)
    rows.append({"idx": m["idx"], "file": os.path.basename(m["gen"]), "seed": m["seed"],
                 "category": m["category"], "recall": round(rec, 3), "missing": miss, "asr": hyp})
    mark = "OK " if rec >= 0.999 else "!! "
    if rec < 0.999:
        print(f"{mark}{m['idx']:02d} {os.path.basename(m['gen']):<32} recall={rec:.2f} missing={miss}")
    sys.stdout.flush()

json.dump(rows, open(os.path.join(D, "asr_gigaam.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
lines = [f"{r['idx']:02d} | {r['file']} | {r['recall']} | {','.join(r['missing'])} | {r['asr'][:70]}"
         for r in rows]
open(os.path.join(D, "asr_gigaam_table.txt"), "w", encoding="utf-8").write("\n".join(lines))
low = [r for r in rows if r["recall"] < 0.999]
print(f"\nTOTAL: {len(rows)} files, recall<1.0: {len(low)}")
import statistics as st
print("mean recall:", round(st.mean(r["recall"] for r in rows), 4))
