# -*- coding: utf-8 -*-
"""Классификация clean-diction клипов по ШЕСТИ корзинам фонетических контрастов (S16).

Целевой объект — фонетический КОНТРАСТ/семейство, а не конкретная строка или слово.
Корзины (по решению пользователя 20.09):
  1. yi           — ы↔и (минимальные пары, «мы/ми», «был/бил», «ты/ти»)
  2. softness     — твёрдость/мягкость и палатализация (ч/ц, ш/щ, л/ль, н/нь перед и/е/ё/ю/я)
  3. clusters     — сложные стечения согласных (ств/нств/рств/нкт/ртр/вств/здн/мпл…)
  4. devoicing    — оглушение/редукция с потерей фонем (конечные звонкие, безударные о/а/е)
  5. stress       — ударения: омографы и разноместное ударение (з+амок/зам+ок, атл+ас/атл+ас)
  6. long_num     — длинные/составные слова + числа/даты
  7. topwords     — реальные top-error слова НО в новых формах/контекстах (отдельная корзина)

ВАЖНО: ни один текст из hard_eval_v1 (frozen) не должен попасть в train —
проверка 4-граммами выполняется здесь же.

usage: python phonetic_baskets.py [--out local_train/data_s2_full/v16_diction/baskets.jsonl]
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
CLEAN = os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "clean_clips.jsonl")
FROZEN = os.path.join(AUK, "local_train", "hard_eval_v1", "hard_eval_v1.jsonl")

TOP_WORDS = {"сегодня", "университета", "университет", "тебя", "холодильник", "кажется",
             "сколько", "работает", "целый", "лучше", "уже", "тысяча", "командировку",
             "предыдущего", "зарегистрировались", "ключи", "пять", "учится",
             "автоматически", "встреча", "вторая", "рублей", "петровна", "крыльца", "целый"}

CLUSTERS = re.compile(r"(ств|нств|рств|вств|нкт|нкт|ртр|здн|мпл|нтр|ктр|вск|здк|жч|щн|чт|"
                      r"вствл|скл|ткл|дн[ое]|рмн|лбц|взд|стк|нтк)")

# ы/и: МИНИМАЛЬНЫЕ ПАРЫ и слова, где контраст ы↔и смыслообразующий.
# Обычные местоимения «мы/ты/свои» исключены (встречаются в 44% корпуса → корзина теряет смысл).
YI_PAT = re.compile(r"\b(был|бил|была|била|были|били|рыть|рить|мыть|мил|мыл|"
                    r"пыль|пиль|крыть|крит|крыл|крил|ныть|нит|выть|вить|лыжи|лижи|"
                    r"сыт|сит|рык|рик|тыл|тил|плыв|плыви|мыши|миши|сынов|сын|син|"
                    r"тыкв|тик|ныр|нир|выл|вил|рыл|рил|мыл|мил)\w*\b")
# мягкость/палатализация: КОНТРАСТНЫЕ пары (не любой мягкий согласный — иначе ловит 72% корпуса)
# ч/щ против ц/ш; шипящие+н/т; мягкий знак перед согласным; л/н перед и/е/ё/ю/я в длинных словах
SOFT_PAT = re.compile(r"(чн|чт|чк|щн|щт|шь[еёюяи]|щь|ч[аоуыэ]|щ[аоуыэ]|ц[иеёюя]|"
                      r"[лн][иеёюя]\w{4,}|нч|нщ|рч|рщ)")
# оглушение/редукция: конечные звонкие и безударные гласные в длинных словах
DEVOIC_PAT = re.compile(r"\b\w{6,}(б|в|г|д|ж|з)\b")
# «трудные» суффиксы для корзины stress (ударение в них неочевидно):
# -ировать/-ение/-атель/-итель/-ология/-ография/-метрия/-ичество/-ность/-изм/-ика/-ическ…
TRICKY_SUF = re.compile(r"(ировать|ирует|ировал|ение|ения|ению|атель|итель|ология|ография|"
                        r"метрия|ичество|ность|изм\b|ика\b|ика[.,!?]|ическ)")
# числа/даты
NUM_PAT = re.compile(r"\b\d+\b|\b(тысяч|миллион|миллиард|процент|градус|рубл|доллар|евро|"
                     r"январ|феврал|марта|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр)"
                     r"\w*\b")


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "").replace("ё", "е")).strip()


def fourgrams(t):
    w = t.split()
    return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))} if len(w) >= 4 else set()


def baskets(text):
    t = (text or "").lower()
    out = set()
    if YI_PAT.search(t):
        out.add("yi")
    if SOFT_PAT.search(t):
        out.add("softness")
    if CLUSTERS.search(t):
        out.add("clusters")
    if DEVOIC_PAT.search(t) or re.search(r"\b(хлеб|дуб|гроб|сугроб|гриб|зуб|лоб|снег|бег|лёд|мёд|"
                                         r"род|народ|город|порог|сапог|пирог|друг|круг|луг|флаг|враг|шаг|"
                                         r"мороз|глаз|рассказ|указ|приказ|вокзал|мост|гвоздь|князь|грязь|"
                                         r"связь|суть|грудь|просьба|резьба|борьба|мольба|свадьба|лодка|шапка|"
                                         r"папка|трубка|шубка|книжка|малышка|рубашка|подружка)\w*\b", t):
        out.add("devoicing")
    if NUM_PAT.search(t):
        out.add("long_num")
    words = re.findall(r"[а-яё]+", t)
    if any(len(w) >= 10 for w in words):
        out.add("long_num")
    if set(words) & TOP_WORDS:
        out.add("topwords")
    # stress: омографы ИЛИ «трудные» суффиксы (>=2 на фразу) + 2 длинных слова.
    # Омографов в читаной речи мало (267 клипов); суффиксный детектор даёт 11-12% пула
    # (ПРОВЕРЕНО 21.09 на read-пуле 22272 клипа).
    if re.search(r"\b(замок|замки|атлас|атласы|мука|муки|стрелки|стрелки|плачу|плачу|"
                 r"дорога|дорого|окна|окно|пила|пила|стоит|стоят|начал|начала|понял|поняла)\b", t):
        out.add("stress")
    elif len(TRICKY_SUF.findall(t)) >= 2 and len(re.findall(r"\b[а-яё]{10,}\b", t)) >= 2:
        out.add("stress")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(AUK, "local_train", "data_s2_full", "v16_diction", "baskets.jsonl"))
    args = ap.parse_args()

    frozen = [json.loads(l) for l in open(FROZEN, encoding="utf-8") if l.strip()]
    frozen_4g = set()
    for r in frozen:
        frozen_4g |= fourgrams(norm(r["text"]))
    print(f"frozen 4-grams: {len(frozen_4g)}")

    clips = [json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip()]
    print("clean clips:", len(clips))

    rows = []
    counter = Counter()
    per_basket = defaultdict(list)
    excluded = 0
    for c in clips:
        t = norm(c["text"])
        if fourgrams(t) & frozen_4g:
            excluded += 1
            continue  # защита: не включать даже похожее на frozen
        bs = baskets(c["text"])
        if not bs:
            bs = {"general"}
        for b in bs:
            counter[b] += 1
            per_basket[b].append(c["path"])
        rows.append({"path": c["path"], "text": c["text"], "dur": c["dur"],
                     "baskets": sorted(bs), "hard_words": c.get("hard_words") or [],
                     "wer": c.get("wer")})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"classified: {len(rows)} clips | excluded (4-gram overlap with frozen): {excluded}")
    print("basket coverage:")
    for b in ("yi", "softness", "clusters", "devoicing", "stress", "long_num", "topwords", "general"):
        n = counter.get(b, 0)
        hrs = sum(r["dur"] for r in rows if b in r["baskets"]) / 3600
        print(f"  {b:12s}: {n:5d} clips  {hrs:5.2f} h")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
