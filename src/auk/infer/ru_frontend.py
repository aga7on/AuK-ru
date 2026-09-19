# -*- coding: utf-8 -*-
"""Russian frontend (S9): raw text → speakable text для AuK-ru.

Конвейер:
    raw text
      → rutextnorm.normalize_russian (числа, даты, валюты, телефоны, URL, аббревиатуры, единицы)
      → _accentize_ru (RUAccent + pron-лексикон: ударения, омографы, ё)
      → (опционально) ru_to_latin (транслитерация для s0-инференса)

Использование в инференсе:
    from auk.infer.ru_frontend import to_speakable
    text = to_speakable("Встреча 21.09.2026 в 14:30")

Диагностика: flag_uncertain() из rutextnorm — span'ы, которые правила не разрешили
(кандидаты на ручной/LLM-разбор).
"""
import os
import sys

_LOCAL_RTN = os.path.join(os.path.dirname(__file__), "..", "..", "..", "local_train", "thirdparty", "rutextnorm")
if os.path.isdir(_LOCAL_RTN):
    sys.path.insert(0, os.path.abspath(_LOCAL_RTN))


def normalize_numbers(text: str) -> str:
    """Числа/даты/валюты/аббревиатуры → слова (rutextnorm)."""
    from rutextnorm import normalize_russian
    return normalize_russian(text)


def flag_uncertain(text: str):
    """Span'ы, которые rutextnorm не уверенно разрешил."""
    from rutextnorm import flag_uncertain as _fu
    return _fu(text)


_WORD_RE = None


def _letters(w: str) -> str:
    """Слово без маркеров ударения, lower."""
    return w.replace("+", "").lower()


def _equiv(a: str, b: str) -> bool:
    """Эквивалентность с учётом легитимной ё-расстановки (е→ё)."""
    return a.replace("ё", "е") == b.replace("ё", "е")


def safe_accentize(text: str, use_lexicon: bool = True) -> str:
    """_accentize_ru с защитой от фонетического респеллинга (S9.1).

    Инвариант: акцентизация только ДОБАВЛЯЕТ маркеры «+» (и легитимно е→ё).
    Если RUAccent заменил буквы слова (напр. «Встреча»→«Фстреча», оглушение) —
    слово возвращается в исходном виде без акцентов. Модель AuK-ru обучена на
    орфографическом тексте; фонетическая орфография деградирует дикцию
    (ПРОВЕРЕНО: frontend_probe fb_020, S9_FRONTEND.md находка 1).
    """
    global _WORD_RE
    import re as _re
    if _WORD_RE is None:
        _WORD_RE = _re.compile(r"\S+|\s+")
    from auk.infer.infer_gradio import _accentize_ru
    acc = _accentize_ru(text, use_lexicon=use_lexicon)
    src_tokens = _WORD_RE.findall(text)
    acc_tokens = _WORD_RE.findall(acc)
    if len(src_tokens) != len(acc_tokens):
        # структура разошлась — безопасный откат: акцентируем только если буквы целы глобально
        return acc if _equiv(_letters(text), _letters(acc)) else text
    out = []
    for s, a in zip(src_tokens, acc_tokens):
        if _equiv(_letters(s), _letters(a)):
            out.append(a)
        else:
            out.append(s)  # респеллинг — откат к исходному слову
    return "".join(out)


def _tech_prespell(text: str) -> str:
    """Техно-домен → проговариваемые русские конструкции (S9, находка 2/3).

    ПРОВЕРЕНО пробой: побуквенные цепочки («хттпс двоеточие слэш») и дефисные
    телефонные группы модель не произносит. Здесь — консервативные замены
    ДО rutextnorm: URL/email/версии/«Тел.».
    """
    import re as _re
    # email: mail@example.com → «мейл собачка пример точка ком»
    text = _re.sub(
        r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})",
        lambda m: f"{m.group(1)} собачка {m.group(2)} точка {m.group(3)}",
        text)
    # URL: scheme://domain/path?query → «сайт domain точка tld» (путь отбрасываем)
    text = _re.sub(
        r"https?://([A-Za-z0-9.-]+)\.[A-Za-z]{2,}(?:/[^\s]*)?",
        lambda m: f"сайт {m.group(1)} точка {_TLD.get(m.group(0).rsplit('.', 1)[-1].split('/')[0], 'ком')}",
        text)
    text = _re.sub(r"www\.([A-Za-z0-9.-]+)", lambda m: f"сайт {m.group(1)}", text)
    # версии: v2.0.1 → «версии два ноль один»
    text = _re.sub(r"\bv(\d+)\.(\d+)(?:\.(\d+))?",
                   lambda m: f"версии {m.group(1)} {m.group(2)}" + (f" {m.group(3)}" if m.group(3) else ""),
                   text)
    # «Тел.» → «Телефон» (rutextnorm не раскрывает, модель читает «тел» как «делал»)
    text = _re.sub(r"\bТел\.", "Телефон", text, flags=_re.IGNORECASE)
    # дефисы в телефонных группах → пробелы (8-800-555 → 8 800 555): чище дикция
    text = _re.sub(r"(?<=\d)-(?=\d)", " ", text)
    # частые аббревиатуры → побуквенно/полно (ПРОВЕРЕНО v2 probe: «ООО»→«алло», «МГУ»→«юмгум»)
    for item in _ABBR:
        pat, repl = item[0], item[1]
        text = _re.sub(pat, repl, text)
    return text


_ABBR = [
    # Форма записи выбрана по abbr_form_probe (16 генераций, s7@5750):
    # «точки» — единственная форма, где побуквенное чтение стабильно («мгу им ломоносова»,
    # WER 0.50/0.62); пробелы сливаются в «мг уим» (0.57), raw → «мангу» (0.60/0.80).
    # Для ООО все формы провалились (0.75–1.0) — оставлено «о.о.о.» до лексикона/модельной правки.
    (r"\bООО\b", "о.о.о."),
    (r"\bЗАО\b", "з.а.о."),
    (r"\bОАО\b", "о.а.о."),
    (r"\bПАО\b", "п.а.о."),
    (r"\bИП\b", "ип"),
    (r"\bМГУ\b", "м.г.у."),
    (r"\bСПбГУ\b", "с.п.б.г.у."),
    (r"\bВШЭ\b", "в.ш.э."),
    (r"\bМФТИ\b", "м.ф.т.и."),
    (r"\bНИИ\b", "научно исследовательский институт"),
    (r"\bООН\b", "о.о.н."),
    (r"\bНАТО\b", "нато"),
    (r"\bВОЗ\b", "в.о.з."),
    (r"\bЦБ РФ\b", "центральный банк России"),
    (r"\bРФ\b", "Российской Федерации"),
    (r"\bРЖД\b", "р.ж.д."),
    (r"\bМВД\b", "м.в.д."),
    (r"\bФСИН\b", "фсин"),
    (r"\bЖКХ\b", "ж.к.х."),
    (r"\bЗАГС\b", "загс"),
    (r"\bЕГЭ\b", "е.г.э."),
    (r"\bДМС\b", "д.м.с."),
    (r"\bОСАГО\b", "осаго"),
    (r"\bКАСКО\b", "каско"),
    (r"\bГОСТ\b", "гост"),
    (r"\bСНИЛС\b", "снилс"),
    (r"\bИНН\b", "инн"),
    (r"\bТЦ\b", "торговый центр"),
    (r"\bREADME\b", "ридми"),
    (r"\bnpm\b", "н.п.м."),
    (r"\bAPI\b", "а.п.и."),
    (r"\bGPU\b", "г.п.ю."),
    (r"\bCPU\b", "с.п.ю."),
    (r"\bSSH\b", "с.с.х."),
    (r"\bCI\b", "с.и."),
]


_TLD = {"ru": "ру", "com": "ком", "org": "орг", "net": "нет", "io": "ио"}


def to_speakable(text: str, accentize: bool = True, translit: bool = False) -> str:
    """Полный frontend: tech-prespell → normalize → safe accentize → (translit)."""
    out = normalize_numbers(_tech_prespell(text))
    if accentize:
        out = safe_accentize(out)
    if translit:
        from auk.infer.ru_translit import ru_to_latin
        out = ru_to_latin(out)
    return out
