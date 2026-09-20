# -*- coding: utf-8 -*-
"""Скан полного корпуса на ДЕФИЦИТНЫЕ корзины (stress, devoicing) — тексты без аудио-чтения.

Источник: local_train/corpus/corpus_index.jsonl (весь корпус ru_wav_mfa, ~145 МБ индекса).
Цель: найти клипы, содержащие омографы/ударные контрасты и оглушение-паттерны, чтобы
закрыть дыры S16-корзин (stress 0.06 h, devoicing 0.49 h).

Выход: local_train/data_s2_full/v16_diction/deficit_candidates.jsonl
(кандидаты для последующей ASR-верификации; НЕ train-строки напрямую).
"""
import json
import os
import re
from collections import Counter

AUK = r"G:\AI\AuK"
INDEX = os.path.join(AUK, "local_train", "corpus", "corpus_index.jsonl")
FROZEN = os.path.join(AUK, "local_train", "hard_eval_v1", "hard_eval_v1.jsonl")
EXISTING = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "clean_clips.jsonl")
OUT = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "deficit_candidates.jsonl")

# --- корзина STRESS: омографы и разноместное ударение (слова, где ударение меняет смысл) ---
OMOGRAPHS = [
    "замок", "замки", "замка", "атлас", "атласы", "мука", "муки", "муку",
    "стрелки", "стрелок", "пила", "пилы", "пилу", "стоит", "стоят", "стоила",
    "начал", "начала", "начали", "понял", "поняла", "поняли", "занимал", "занимала",
    "отдал", "отдала", "прожил", "прожила", "была", "были", "взяла", "взяли",
    "ждала", "ждали", "рвал", "рвала", "звала", "звали", "гнала", "гнали",
    "дорога", "дорого", "дороги", "окна", "окон", "окно", "сверло", "сверла",
    "весел", "весела", "творог", "творога", "договор", "договора", "договоры",
    "каталог", "каталога", "звонит", "звонишь", "звонят", "включит", "включишь",
    "облегчить", "облегчит", "красивее", "красивей", "торты", "тортов", "банты",
    "бантов", "шарфы", "шарфов", "щавель", "щавеля", "сливовый", "свекла", "свеклы",
    "дефис", "дефиса", "диспансер", "диспансера", "документ", "документа", "документы",
    "километр", "километра", "сантиметр", "сантиметра", "кухонный", "кухонного",
    "одновременно", "квартал", "квартала", "намерение", "намерения", "осужден",
    "призрак", "призрака", "сироты", "сирот", "сливовый", "средства", "средств",
    "столовая", "столовой", "украинский", "цемента", "черпать", "эксперт", "эксперта",
]

# --- корзина DEVOICING/редукция: слова с оглушением конечных/срединных звонких ---
DEVOICE_WORDS = [
    "хлеб", "дуб", "гроб", "сугроб", "гриб", "грибы", "зуб", "зубы", "лоб", "лбы",
    "снег", "снега", "бег", "бега", "лёд", "льда", "мёд", "мёда", "род", "рода",
    "народ", "народа", "город", "города", "порог", "порога", "сапог", "сапоги",
    "пирог", "пироги", "друг", "друзья", "круг", "круги", "луг", "луга", "вдруг",
    "флаг", "флаги", "враг", "враги", "шаг", "шаги", "залог", "итоги", "дорожка",
    "ножка", "ложка", "повтор", "уговор", "двор", "ковёр", "забор", "топор",
    "мороз", "морозы", "глаз", "глаза", "рассказ", "рассказы", "указ", "приказ",
    "вокзал", "вокзаля", "мосты", "мост", "гвоздь", "гвозди", "князь", "грязь",
    "связь", "связи", "суть", "грудь", "обуть", "сбить", "просьба", "резьба",
    "борьба", "мольба", "свадьба", "лодка", "лодки", "шапка", "шапки", "папка",
    "трубка", "шубка", "книжка", "малышка", "девчонка", "рубашка", "подружка",
    "бумага", "бумаги", "дорогой", "дороговато", "молодой", "седой", "босой",
]

OMO_SET = set(OMOGRAPHS)
DEV_SET = set(DEVOICE_WORDS)

# --- корзина YI: минимальные пары ы↔и (контраст смыслообразующий) ---
YI_WORDS = [
    "был", "бил", "была", "била", "были", "били", "рыть", "рить", "мыть", "мил", "мыл",
    "пыль", "пиль", "крыть", "крит", "крыл", "крил", "ныть", "нит", "выть", "вить",
    "лыжи", "лижи", "сыт", "сит", "рык", "рик", "тыл", "тил", "мыши", "сын", "син",
    "тыква", "нырять", "выл", "вил", "рыл", "рил", "мыла", "мила", "рыба", "глыба",
    "плыть", "плить", "мыло", "мило", "дышит", "пишет", "рыщет", "рынок", "синок",
    "мышка", "мишка", "мышцы", "тысяча", "высокий", "высота", "ивы", "ива", "сливы",
]
YI_SET = set(YI_WORDS)

# --- корзина STRESS (расширенная): омографы + длинные слова (≥9 букв), где ударение
#    неочевидно. Порог 9 (не 7) — иначе корзина теряет смысл (ловит половину корпуса).
LONG_STRESS = re.compile(r"\b[а-яё]{9,}\b")


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "").replace("ё", "е")).strip()


def fourgrams(t):
    w = t.split()
    return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))} if len(w) >= 4 else set()


def main():
    frozen_4g = set()
    for l in open(FROZEN, encoding="utf-8"):
        r = json.loads(l)
        frozen_4g |= fourgrams(norm(r["text"]))

    existing = set()
    for src in (EXISTING, os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "deficit_clean.jsonl")):
        if os.path.exists(src):
            for l in open(src, encoding="utf-8"):
                existing.add(json.loads(l)["path"])
    print("already verified/existing:", len(existing))

    cnt = Counter()
    out = []
    for l in open(INDEX, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        p, t = r.get("p"), r.get("t")
        if not p or not t or p in existing:
            continue
        d = float(r.get("d") or 0)
        if not (2.0 <= d <= 15.0):
            continue
        words = re.findall(r"[а-яё]+", t.lower())
        if fourgrams(norm(t)) & frozen_4g:
            continue
        bs = set()
        if any(w in OMO_SET for w in words):
            bs.add("stress")
        if any(w in DEV_SET for w in words):
            bs.add("devoicing")
        if any(w in YI_SET for w in words):
            bs.add("yi")
        if not bs:
            continue
        cnt.update(bs)
        out.append({"path": p, "text": t, "dur": d, "deficit_baskets": sorted(bs)})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("deficit candidates:", len(out))
    print("basket hits:", dict(cnt))
    hours = sum(r["dur"] for r in out) / 3600
    print("hours: %.2f" % hours)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
