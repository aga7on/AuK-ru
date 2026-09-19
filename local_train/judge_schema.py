"""Strict mode-specific schema for the Gemini audit judge (v3).

status 'ok' is allowed only if validate_strict returns no problems. Lists and bools are
required and type-checked; scores must be ints 0-10; verdict must be a valid enum value.
"""
VERDICTS = {"годен", "доработка", "брак"}

COMMON = {"audio_access": bool, "overall": int, "verdict": str}
TTS = {"text_fidelity": int, "endings": int, "naturalness": int, "prosody": int, "accent": int,
       "palatalization": int, "stress": int, "artifacts": int, "voice_similarity_to_ref": int,
       "truncated": bool, "words_mangled": list, "words_dropped": list,
       "words_misstressed": list, "phoneme_substitutions": list, "issues": str}
CLONE = {"text_fidelity": int, "voice_similarity_to_ref": int, "naturalness": int, "artifacts": int,
         "ref_content_leak": bool, "leaked_ref_content": list, "words_mangled": list,
         "words_dropped": list, "issues": str}
OP = {"operation_performed": int, "content_preserved": int, "quality": int,
      "expected_effect_matches": bool, "introduced_artifacts": bool, "heard_result": str, "issues": str}

TTS_NOREF = {k: v for k, v in TTS.items() if k != "voice_similarity_to_ref"}

SCHEMAS = {"tts": TTS, "phonetics": TTS, "phrase_training": TTS, "clone": CLONE,
           "tool": OP, "capability": OP}
INT_SCORES = {"overall", "text_fidelity", "endings", "naturalness", "prosody", "accent",
              "palatalization", "stress", "artifacts", "voice_similarity_to_ref",
              "operation_performed", "content_preserved", "quality"}


def validate_strict(group, obj, has_ref=True):
    problems = []
    if not isinstance(obj, dict):
        return ["not an object"]
    schema = SCHEMAS.get(group)
    if schema is None:
        return [f"unknown group {group!r}"]
    if group in ("tts", "phonetics", "phrase_training") and not has_ref:
        schema = TTS_NOREF
    for key, typ in {**COMMON, **schema}.items():
        if key not in obj:
            problems.append(f"missing {key}")
            continue
        v = obj[key]
        if typ is int:
            if isinstance(v, bool) or not isinstance(v, int):
                problems.append(f"{key} not int")
            elif not (0 <= v <= 10):
                problems.append(f"{key} out of range {v}")
        elif typ is bool:
            if not isinstance(v, bool):
                problems.append(f"{key} not bool")
        elif typ is list:
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                problems.append(f"{key} not list[str]")
        elif typ is str:
            if not isinstance(v, str):
                problems.append(f"{key} not str")
    if obj.get("verdict") not in VERDICTS:
        problems.append(f"bad verdict {obj.get('verdict')!r}")
    return problems


def is_ok(group, obj):
    return not validate_strict(group, obj)
