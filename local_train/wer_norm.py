# -*- coding: utf-8 -*-
"""WER-нормализация для frontend-гейтов: приводит expected и heard к общей форме.

Зачем (ПРОВЕРЕНО на v3/v4 probe): ASR (GigaAM) транскрибирует побуквенные цепочки
слитно («о.о.о.» → «ооо», «а.п.и.» → «апи»), из-за чего честное произношение
получает WER 0.56–0.8. Нормализация убирает точки/пробелы между одиночными
кириллическими/латинскими буквами и приводит обе стороны к одному виду.
"""
import re
import sys
import unicodedata

sys.path.insert(0, r"G:\AI\AuK\local_train")

SINGLE_LETTERS = re.compile(r"\b([A-Za-zА-Яа-яЁё])\.(?=\s|\b)")


def collapse_spelled(text: str) -> str:
    """«о.о.о.» → «ооо»; «м.г.у.» → «мгу»; «а.п.и.» → «апи»."""
    prev = None
    cur = text
    # повторяем, пока есть цепочки вида X. Y.
    for _ in range(6):
        if cur == prev:
            break
        prev = cur
        cur = re.sub(r"\b([A-Za-zА-Яа-яЁё])\.\s*(?=[A-Za-zА-Яа-яЁё]\.)", r"\1", cur)
        cur = re.sub(r"\b([A-Za-zА-Яа-яЁё])\.(?=\s|$|[^\w])", r"\1", cur)
    # добиваем одиночные буквы с точкой
    cur = re.sub(r"\b([A-Za-zА-Яа-яЁё])\.", r"\1", cur)
    return cur


def norm_for_wer(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("«", "").replace("»", "").replace('"', "")
    t = collapse_spelled(t)
    t = t.replace("ё", "е").replace("+", "")
    t = re.sub(r"[^0-9a-zа-я ]+", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t
