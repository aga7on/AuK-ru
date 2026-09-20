# -*- coding: utf-8 -*-
"""S13b: манифест судьи v3 (group=phonetics) для ВСЕХ 80 файлов слепого A/B.

Цель: объективно подтвердить/опровергнуть наблюдения пользователя о русском произношении —
«клучи» вместо «ключи» (палатализация/подмена), «ми» вместо «мы» (ы→и), потеря ударений.
Схема phonetics (judge_schema.TTS) содержит именно эти оси: text_fidelity, endings,
accent, palatalization, stress, phoneme_substitutions, words_mangled/dropped.

Референс не передаётся → схема TTS_NOREF (оценивается только произношение по тексту),
что и нужно: нас интересует дикция, а не похожесть голоса.

Целевой текст = to_speakable(pair_text): числа/даты раскрыты (иначе судья ложно бракует
«1423» против «одна тысяча четыреста двадцать три»), ударения помечены «+» — судья
знает ожидаемое ударение (конвенция промпта v3).

usage: python s13_phonetics_manifest.py [--out <manifest.json>]
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
PAIR_DIR = os.path.join(AUK, "local_tests", "pairwise_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        AUK, "local_train", "reports", "deepseek_supervised", "s13_phonetics_manifest.json"))
    args = ap.parse_args()

    from auk.infer.ru_frontend import to_speakable

    listen = list(csv.DictReader(open(os.path.join(PAIR_DIR, "LISTEN.csv"), encoding="utf-8")))
    secret = {r["id"]: r for r in csv.DictReader(
        open(os.path.join(PAIR_DIR, "SECRET_pair_map.csv"), encoding="utf-8"))}

    out = []
    miss = 0
    for p in listen:
        pid = p["id"]
        s = secret.get(pid, {})
        try:
            speak = to_speakable(p["text"])
        except Exception:
            speak = p["text"]
        for side in ("a", "b"):
            f = os.path.join(PAIR_DIR, p[f"{side}_wav"])
            if not os.path.exists(f):
                miss += 1
                continue
            variant = s.get(f"{side}_variant", "?")
            out.append({
                "id": f"{pid}_{side}_{variant}",
                "mode": "tts",          # mode выбирает ПРОМПТ (tts/clone/op); group — схему валидации
                "group": "phonetics",
                "text": speak,
                "ref": None,
                "file": f,
                "wav": f,
                "goal": "Чистое русское произношение: верные ударения, палатализация, окончания",
                "checks": ("русское произношение без иностранного акцента; мягкие согласные перед и/е/ё/ю/я "
                           "произнесены мягко (НЕ «клучи» вместо «ключи»); «ы» не заменяется на «и» "
                           "(НЕ «ми» вместо «мы»); ударения на правильных слогах; окончания не проглочены"),
            })

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} (missing wav: {miss}) -> {args.out}")


if __name__ == "__main__":
    main()
