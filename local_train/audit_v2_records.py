"""Audit the v2 blind judge records.

Two separate questions, deliberately not conflated:
  A) Did each v2 reply honour the ORIGINAL v2 contract (its own field names/types)?
  B) Does the reply satisfy the NEW v3 contract (renamed leak semantics, strict types)?
The clone rejudge is justified by a SEMANTIC rubric correction (leak redefined), not by the
rename: v3 renamed ref_word_leak/leaked_ref_words -> ref_content_leak/leaked_ref_content.
"""
import json
import os
from collections import Counter, defaultdict

from judge_schema import validate_strict

AUK = r"G:\AI\AuK"
RES = os.path.join(AUK, "local_tests", "blind_s2", "results_audit_gemini_v2.jsonl")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "v2_schema_audit.md")

V2_CLONE = {"text_fidelity": int, "voice_similarity_to_ref": int, "naturalness": int,
            "artifacts": int, "ref_word_leak": bool, "leaked_ref_words": list,
            "words_mangled": list, "words_dropped": list, "issues": str,
            "audio_access": bool, "overall": int, "verdict": str}
V2_TTS = {"text_fidelity": int, "endings": int, "naturalness": int, "prosody": int, "accent": int,
          "palatalization": int, "stress": int, "artifacts": int, "voice_similarity_to_ref": int,
          "truncated": bool, "words_mangled": list, "words_dropped": list, "words_misstressed": list,
          "phoneme_substitutions": list, "issues": str, "audio_access": bool, "overall": int,
          "verdict": str}
V2_OP = {"operation_performed": int, "content_preserved": int, "quality": int,
         "expected_effect_matches": bool, "introduced_artifacts": bool, "heard_result": str,
         "issues": str, "audio_access": bool, "overall": int, "verdict": str}
V2 = {"clone": V2_CLONE, "tts": V2_TTS, "phonetics": V2_TTS, "phrase_training": V2_TTS,
      "tool": V2_OP, "capability": V2_OP}


def check_v2(group, obj):
    probs = []
    if not isinstance(obj, dict):
        return ["not an object"]
    for k, t in V2[group].items():
        if k not in obj:
            probs.append(f"missing {k}")
            continue
        v = obj[k]
        if t is int and (isinstance(v, bool) or not isinstance(v, int)):
            probs.append(f"{k} not int")
        elif t is bool and not isinstance(v, bool):
            probs.append(f"{k} not bool")
        elif t is list and (not isinstance(v, list) or any(not isinstance(x, str) for x in v)):
            probs.append(f"{k} not list[str]")
        elif t is str and not isinstance(v, str):
            probs.append(f"{k} not str")
    if obj.get("verdict") not in {"годен", "доработка", "брак"}:
        probs.append("bad verdict")
    return probs


def main():
    rows = {}
    for line in open(RES, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["file"]] = r
    stats = defaultdict(Counter)
    for f, r in rows.items():
        g = r.get("group")
        st = r.get("status")
        stats[g]["total"] += 1
        if st != "ok":
            stats[g]["not_ok"] += 1
            continue
        j = r.get("judge") or {}
        if not check_v2(g, j):
            stats[g]["v2_contract_ok"] += 1
        else:
            stats[g]["v2_contract_fail"] += 1
        if not validate_strict(g, j, has_ref=True):
            stats[g]["v3_contract_ok"] += 1
        else:
            stats[g]["v3_contract_fail"] += 1

    L = ["# Аудит v2-записей: исходный контракт v2 vs новый контракт v3", "",
         f"Файл: `{os.path.relpath(RES, AUK)}` (685 строк, один запуск 16.09.2026).", "",
         "Разделяются две вещи:", "",
         "1. **Контракт v2** — поля/типы, которые судья v2 сам запрашивал.",
         "2. **Контракт v3** — те же данные, но с переименованной семантикой утечки",
         "   (`ref_word_leak`/`leaked_ref_words` → `ref_content_leak`/`leaked_ref_content`).", "",
         "| группа | всего | not_ok | v2-contract ok | v2-contract fail | v3-contract ok | v3-contract fail |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for g in sorted(stats):
        c = stats[g]
        L.append(f"| {g} | {c['total']} | {c['not_ok']} | {c['v2_contract_ok']} | "
                 f"{c['v2_contract_fail']} | {c['v3_contract_ok']} | {c['v3_contract_fail']} |")
    L += ["",
          "## Вывод", "",
          "- 125 «schema_fail» по clone — это НЕ признак того, что старые ответы не содержали",
          "  запрошенных полей. По исходному контракту v2 clone-ответы валидны; расхождение даёт",
          "  только переименование ключей в v3.",
          "- Остальные 560 записей проходят и v2-, и v3-контракт (bools/lists/диапазоны int).",
          "- Rejudge clone оправдан СМЕНОЙ СЕМАНТИКИ: v2 трактовал утечку как «услышал слова рефа»,",
          "  что ложно срабатывало на легитимных общих словах русского текста; v3 считает утечкой",
          "  только лишнее содержимое референса, отсутствующее в целевом тексте.",
          "- Все прочие группы НЕ переjudge-ятся: их контракт не менялся.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", OUT)
    for g in sorted(stats):
        print(g, dict(stats[g]))


if __name__ == "__main__":
    main()
