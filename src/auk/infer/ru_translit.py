"""Russian -> Latin transliteration for English-centric TTS models (AuK).

Cyrillic runs are rewritten in Latin letters so the acoustic model reads
Russian words with clear, correct phonetics instead of mangled Cyrillic.
A combining acute accent (U+0301) after a vowel is rendered as an apostrophe
after that vowel to hint the stress position.

Example:
    "Приве́т! Это прове́рка." -> "Prive't! Eto prove'rka."
"""
from __future__ import annotations

_ACUTE = "\u0301"

_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "'", "ы": "y", "ь": "'", "э": "e", "ю": "yu", "я": "ya",
}


def _convert_char(ch: str) -> str:
    mapped = _MAP.get(ch.lower())
    if mapped is None:
        return ch
    if ch.isupper() and mapped:
        return mapped.capitalize()
    return mapped


def ru_to_latin(text: str) -> str:
    """Transliterate every Cyrillic character in *text*; other characters pass through.

    Stress hints are accepted in two notations and rendered as an apostrophe
    right after the stressed vowel:
      * ``+`` before the vowel (RuAccent output, e.g. ``пров+ерка``)
      * combining acute accent U+0301 after the vowel (``прове́рка``)
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "+" and i + 1 < len(text) and text[i + 1].lower() in _MAP:
            out.append(_convert_char(text[i + 1]) + "'")
            i += 2
            continue
        if i + 1 < len(text) and text[i + 1] == _ACUTE and ch.lower() in _MAP:
            out.append(_convert_char(ch) + "'")
            i += 2
            continue
        out.append(_convert_char(ch))
        i += 1
    return "".join(out)


if __name__ == "__main__":
    import sys
    print(ru_to_latin(sys.argv[1] if len(sys.argv) > 1 else "Приве́т! Это прове́рка."))
