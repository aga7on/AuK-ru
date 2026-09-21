# -*- coding: utf-8 -*-
"""Трассировка релизных семплов: какой текст/инструкция/референс их породил.

Ищет тексты семплов (по ASR) среди результатов эмо-тестов, чтобы честно подписать:
какая эмоциональная инструкция задавалась и какой референс использовался.

usage: python trace_samples_origin.py
"""
import json
import os
import re
import sys

AUK = r"G:\AI\AuK"
FACTS = os.path.join(AUK, "local_train", "tmp_samples_check", "samples_facts.json")
SEARCH_DIRS = ["emotion_ru_s7_5750", "emotion_ru_s7_6750", "emotion_ru",
               "emotion_ru_s8_7750", "emotion_ru_s8b_8500", "emotion_ru_s16", "emotion_ru_s16c"]


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("ё", "е")).strip()


def main():
    facts = json.load(open(FACTS, encoding="utf-8"))
    heard = {f["file"]: norm(f["heard"]) for f in facts}
    print("samples heard texts:")
    for k, v in heard.items():
        print("  ", k, "->", v)

    idx = []
    for d in SEARCH_DIRS:
        p = os.path.join(AUK, "local_tests", d, "results.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            idx.append({"src": d, "id": r.get("id"), "voice": r.get("voice"),
                        "emotion": r.get("emotion"), "text": norm(r.get("text")),
                        "raw_text": r.get("text"), "ref": r.get("ref"),
                        "file": os.path.basename(r.get("file") or ""), "seed": r.get("seed")})
    print("indexed test rows:", len(idx))

    out = []
    for fn, h in heard.items():
        hits = [r for r in idx if r["text"] == h]
        out.append({"sample": fn, "heard": h, "matches": hits[:6]})
        print(f"\n{fn}: matches={len(hits)}")
        for r in hits[:4]:
            print("   ", r["src"], r["id"], "| voice:", r["voice"], "| emo:", r["emotion"],
                  "| seed:", r["seed"], "| file:", r["file"])
            print("      ref:", r["ref"])

    json.dump(out, open(os.path.join(AUK, "local_train", "tmp_samples_check", "origin.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote origin.json")


if __name__ == "__main__":
    main()
