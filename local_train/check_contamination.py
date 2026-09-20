# -*- coding: utf-8 -*-
"""Проверка contamination: тексты hard_cases / eval-пака против train-миксов.

Правило GATES: текст из оценочного набора НЕ должен встречаться в train.jsonl,
иначе first_ok/WER на нём ничего не доказывают.

usage: python check_contamination.py
"""
import json
import os
import re
from collections import Counter

AUK = r"G:\AI\AuK"


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "")).strip()


def load_eval_texts():
    texts = set()
    packs = [
        (os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), None),
        (os.path.join(AUK, "local_tests", "phonetic_pack", "pack.json"), None),
    ]
    for p, _ in packs:
        for e in json.load(open(p, encoding="utf-8")):
            t = e.get("text")
            if not t:
                m = re.search(r"'([^']*)'", e.get("instruction") or "")
                t = m.group(1) if m else None
            if t:
                texts.add(norm(t))
    hc = os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru_v1.jsonl")
    if os.path.exists(hc):
        for l in open(hc, encoding="utf-8"):
            if l.strip():
                texts.add(norm(json.loads(l)["text"]))
    return texts


def train_texts(path):
    out = set()
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for m in r.get("messages", []):
            if m.get("role") != "user":
                continue
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    mm = re.search(r"'([^']*)'", c["text"])
                    if mm:
                        out.add(norm(mm.group(1)))
    return out


def main():
    ev = load_eval_texts()
    print("eval+hardcase unique texts:", len(ev))
    for name in ("v5_s5_mix", "v7_s7_mix", "v8_s8_mix", "v9_s8b_mix", "v12_emo_mix",
                 "v16_diction_mix"):
        p = os.path.join(AUK, "local_train", "data_s2_full", name, "train.jsonl")
        tr = train_texts(p)
        if not tr:
            print(f"{name}: MISSING {p}")
            continue
        inter = ev & tr
        print(f"{name}: train_texts={len(tr)} intersect={len(inter)}")
        for t in sorted(inter)[:10]:
            print("   LEAK:", t[:90])


if __name__ == "__main__":
    main()
