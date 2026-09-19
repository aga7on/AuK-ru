"""ru_metrics.py — единая система измерения русской генерации (шаг 2 плана s2).

Слои:
  1) Объективный: WER/CER (S/D/I), повторы, лишний текст, аудио-флаги
     (пусто/тишина/NaN/клиппинг/паузы/обрывы), явные отказы ASR.
  2) Судейский (опционально): Gemini через локальный мост (antigravity, :8045),
     рубрика произношения; голосование N раз с усреднением.

Правило: ASR — фильтр содержания, НЕ финальный вердикт о мягкости/ударениях/эмоциях.
CLI:
  python ru_metrics.py --wav path --text "..." [--judge] [--votes 2]
  python ru_metrics.py --dir local_tests/xxx [--judge] [--votes 2]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

AUK = r"G:\AI\AuK"
if AUK not in sys.path:
    sys.path.insert(0, os.path.join(AUK, "src"))
if os.path.join(AUK, "local_train") not in sys.path:
    sys.path.insert(0, os.path.join(AUK, "local_train"))

_NUM_WORDS = {
    "ноль", "один", "одна", "одно", "два", "две", "три", "четыре", "пять", "шесть", "семь",
    "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать", "двадцать",
    "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
    "девятьсот", "тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов",
}


_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двух": 2, "двум": 2, "двумя": 2, "трех": 3, "трёх": 3, "тремя": 3, "четырех": 4, "четырёх": 4,
    "пяти": 5, "шести": 6, "семи": 7, "восьми": 8, "девяти": 9, "десяти": 10,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "двадцати": 20, "тридцати": 30, "сорока": 40, "пятидесяти": 50, "шестидесяти": 60,
    "семидесяти": 70, "восьмидесяти": 80, "девяноста": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
    "ста": 100, "двухсот": 200, "трехсот": 300, "четырехсот": 400, "пятисот": 500,
    "шестисот": 600, "семисот": 700, "восьмисот": 800, "девятисот": 900,
}
_THOUSANDS = {"тысяча", "тысячи", "тысяч", "тысячам", "тысячами", "тысяче"}
_MILLIONS = {"миллион", "миллиона", "миллионов", "миллиону", "миллионам", "миллионами", "миллиард", "миллиарда"}


def _num_token_value(tok: str):
    if re.fullmatch(r"[0-9]+", tok):
        try:
            return int(tok)
        except Exception:
            return None
    for table in (_UNITS, _TENS, _HUNDREDS):
        if tok in table:
            return table[tok]
    return None


def _collapse_numbers(tokens: list[str]) -> list[str]:
    """Числовые последовательности → '#<значение>' (двадцать != пятьдесят)."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        v = _num_token_value(tokens[i])
        if v is None:
            out.append(tokens[i])
            i += 1
            continue
        total = 0
        cur = 0
        j = i
        parsed_any = False
        while j < n:
            tok = tokens[j]
            if tok in _THOUSANDS:
                total += (cur if cur else 1) * 1000
                cur = 0
                parsed_any = True
                j += 1
                continue
            if tok in _MILLIONS:
                total = (total + (cur if cur else 1)) * 1_000_000
                cur = 0
                parsed_any = True
                j += 1
                continue
            vv = _num_token_value(tok)
            if vv is None:
                break
            cur += vv
            parsed_any = True
            j += 1
        if parsed_any:
            out.append(f"#{total + cur}")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    deduped = []
    for t in out:
        if t.startswith("#") and deduped and deduped[-1] == t:
            continue
        deduped.append(t)
    return deduped


def normalize_words(text: str, collapse_numbers: bool = True) -> list[str]:
    """Слова для сравнения: lowercase, ё→е, убрать +/', дефис→пробел; числа → #значение."""
    text = str(text).lower().replace("ё", "е").replace("+", "").replace("'", "")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    toks = []
    for w in text.split():
        if collapse_numbers and (re.fullmatch(r"[0-9]+", w) or w in _UNITS or w in _TENS or w in _HUNDREDS
                                 or w in _THOUSANDS or w in _MILLIONS):
            toks.append(w)
        elif re.search(r"[a-zа-я0-9]", w):
            toks.append(w)
    return _collapse_numbers(toks) if collapse_numbers else toks


def _levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align(ref_words: list[str], hyp_words: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein-выравнивание слов; ops: '=' (совпадение), 'S', 'D', 'I'."""
    n, m = len(ref_words), len(hyp_words)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref_words[i - 1] != hyp_words[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref_words[i - 1] != hyp_words[j - 1]):
            op = "=" if ref_words[i - 1] == hyp_words[j - 1] else "S"
            ops.append((op, ref_words[i - 1], hyp_words[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref_words[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp_words[j - 1]))
            j -= 1
    return ops[::-1]


def text_metrics(ref_text: str, hyp_text: str) -> dict:
    rw = normalize_words(ref_text)
    hw = normalize_words(hyp_text)
    ops = align(rw, hw)
    subs = [(r, h) for op, r, h in ops if op == "S"]
    dels = [r for op, r, h in ops if op == "D"]
    inss = [h for op, r, h in ops if op == "I"]
    hits = sum(1 for op, _, _ in ops if op == "=")
    n_ref, n_hyp = len(rw), len(hw)
    wer = (len(subs) + len(dels) + len(inss)) / max(n_ref, 1)
    recall = hits / max(n_ref, 1)
    precision = hits / max(n_hyp, 1)

    ref_chars = re.sub(r"[^а-яa-z0-9]", "", ref_text.lower().replace("ё", "е"))
    hyp_chars = re.sub(r"[^а-яa-z0-9]", "", hyp_text.lower().replace("ё", "е"))
    cer = _levenshtein(ref_chars, hyp_chars) / max(len(ref_chars), 1)

    adjacent_dupes = sum(1 for i in range(1, n_hyp) if hw[i] == hw[i - 1])
    bigrams = Counter(zip(hw, hw[1:])) if n_hyp > 1 else Counter()
    repeated_bigrams = [f"{a} {b}" for (a, b), c in bigrams.items() if c >= 2]

    char_detail = []
    for r, h in subs:
        char_detail.append({"word": r, "heard": h,
                            "char_ops": _char_ops(r, h)})
    return {
        "wer": round(wer, 3), "cer": round(cer, 3), "recall": round(recall, 3),
        "precision": round(precision, 3), "ref_words": n_ref, "hyp_words": n_hyp,
        "hits": hits, "substitutions": subs, "deletions": dels, "insertions": inss,
        "adjacent_dupes": adjacent_dupes, "repeated_bigrams": repeated_bigrams,
        "word_diffs": char_detail,
    }


def _char_ops(ref_w: str, hyp_w: str) -> list[dict]:
    """Char-level Levenshtein-diff внутри пары слов («развитие→разитие» = del:в)."""
    n, m = len(ref_w), len(hyp_w)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref_w[i - 1] != hyp_w[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref_w[i - 1] != hyp_w[j - 1]):
            if ref_w[i - 1] == hyp_w[j - 1]:
                ops.append({"op": "match", "ch": ref_w[i - 1]})
            else:
                ops.append({"op": "sub", "ref": ref_w[i - 1], "hyp": hyp_w[j - 1]})
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append({"op": "del", "ch": ref_w[i - 1]})
            i -= 1
        else:
            ops.append({"op": "ins", "ch": hyp_w[j - 1]})
            j -= 1
    return ops[::-1]


def audio_metrics(wav_path: str) -> dict:
    import soundfile as sf

    x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr if sr else 0.0
    nan_inf = bool(len(x) and (np.isnan(x).any() or np.isinf(x).any()))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clip_ratio = float(np.mean(np.abs(x) >= 0.985)) if len(x) else 0.0

    frame = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    n = 1 + max(0, (len(x) - frame) // hop)
    if n > 0:
        idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
        r = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(r + 1e-9)
        speech = db > (db.max() - 35)
    else:
        speech = np.zeros(1, dtype=bool)
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
    pauses = []
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            d = (i - start) * hop_s
            if d >= 0.15:
                pauses.append(round(d, 3))
            start = None
    speech_ratio = float(speech.mean()) if len(speech) else 0.0
    touches_end = bool(len(speech) and speech[-1])
    touches_start = bool(len(speech) and speech[0])
    empty = dur < 0.25 or speech_ratio < 0.05
    return {
        "duration": round(dur, 3), "peak": round(peak, 3), "clip_ratio": round(clip_ratio, 5),
        "nan_inf": nan_inf, "lead_silence": round(lead, 3), "tail_silence": round(tail, 3),
        "n_pauses": len(pauses), "pauses_total": round(sum(pauses), 3),
        "pause_max": round(max(pauses), 3) if pauses else 0.0,
        "speech_ratio": round(speech_ratio, 3), "empty": empty,
        "touches_start": touches_start, "touches_end": touches_end,
    }


def transcribe_path(wav_path: str) -> tuple[str, str | None]:
    """GigaAM-транскрипция файла. Возвращает (текст, ошибка-строка)."""
    try:
        import torch
        import soundfile as sf
        from auk.infer import quality

        x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        t = torch.from_numpy(x.T.copy())
        return quality.transcribe(t, sr), None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def asr_flags(heard: str) -> list[str]:
    flags = []
    if not heard.strip():
        flags.append("ASR_EMPTY")
        return flags
    letters = sum(1 for c in heard.lower() if "а" <= c <= "я" or c in "ё ")
    if letters / max(len(heard), 1) < 0.7:
        flags.append("ASR_NOISE")
    return flags


def preliminary_verdict(m: dict, a: dict, flags: list[str]) -> str:
    """Строгий фильтр содержания (пороговые константы — стартовые)."""
    if "ASR_EMPTY" in flags or a["nan_inf"] or a["empty"]:
        return "BAD"
    if m["wer"] > 0.35 or a["clip_ratio"] > 0.003:
        return "BAD"
    if (m["wer"] > 0.12 or m["adjacent_dupes"] or m["insertions"] or
            m["repeated_bigrams"] or "ASR_NOISE" in flags or a["touches_end"] or
            a["clip_ratio"] > 0.0002):
        return "REVIEW"
    return "OK"


def judge_audio(wav_path: str, text: str, votes: int = 2, model: str = "gemini-3.8-flash-medium"):
    """Судейский слой через локальный мост. Возвращает (rubric|None, error|None)."""
    try:
        from auto_judge import call_model, merge_votes
        import base64

        b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
        objs = []
        for _ in range(max(1, votes)):
            obj, _raw = call_model(model, b64, text)
            if obj:
                objs.append(obj)
        if not objs:
            return None, "judge: no valid JSON replies"
        return merge_votes(objs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def evaluate(wav_path: str, ref_text: str, judge: bool = False, votes: int = 2) -> dict:
    heard, asr_err = transcribe_path(wav_path)
    m = text_metrics(ref_text, heard)
    a = audio_metrics(wav_path)
    flags = asr_flags(heard)
    if asr_err:
        flags.append("ASR_ERROR")
    if a["clip_ratio"] > 0.0002:
        flags.append(f"CLIP:{a['clip_ratio']:.4f}")
    verdict = preliminary_verdict(m, a, flags)
    row = {
        "file": os.path.basename(wav_path), "text": ref_text, "heard": heard,
        "asr_error": asr_err, "flags": flags, "verdict": verdict,
        "text_metrics": m, "audio": a,
    }
    if judge:
        rubric, jerr = judge_audio(wav_path, ref_text, votes=votes)
        row["judge"] = rubric
        row["judge_error"] = jerr
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text")
    ap.add_argument("--dir")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--votes", type=int, default=2)
    ap.add_argument("--out", default="ru_metrics.json")
    args = ap.parse_args()

    if args.wav:
        row = evaluate(args.wav, args.text or "", judge=args.judge, votes=args.votes)
        print(json.dumps(row, ensure_ascii=False, indent=1))
        return

    if not args.dir:
        ap.error("need --wav or --dir")
    man_path = os.path.join(args.dir, "manifest.json")
    if not os.path.exists(man_path):
        ap.error(f"manifest.json not found in {args.dir}")
    man = json.load(open(man_path, encoding="utf-8"))
    rows = []
    counts = Counter()
    for item in man:
        wav = item.get("file") or item.get("gen")
        text = item.get("text", "")
        if not wav or not os.path.exists(wav):
            rows.append({"file": wav, "verdict": "BAD", "flags": ["FILE_MISSING"]})
            counts["BAD"] += 1
            continue
        row = evaluate(wav, text, judge=args.judge, votes=args.votes)
        rows.append(row)
        counts[row["verdict"]] += 1
        flags = ",".join(row["flags"]) or "-"
        j = (row.get("judge") or {}).get("overall", "?")
        print(f"{row['verdict']:<6} {row['file']:<34} wer={row['text_metrics']['wer']:.2f} "
              f"cer={row['text_metrics']['cer']:.2f} ins={len(row['text_metrics']['insertions'])} "
              f"dupes={row['text_metrics']['adjacent_dupes']} judge={j} flags={flags}", flush=True)
        if row.get("judge_error"):
            print(f"       judge_error: {row['judge_error']}", flush=True)

    out_path = os.path.join(args.dir, args.out)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nRU_METRICS_DONE {dict(counts)} -> {out_path}")


if __name__ == "__main__":
    main()
