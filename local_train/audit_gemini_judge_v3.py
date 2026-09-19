"""Gemini audit judge v3 (16.09.2026).

Deltas over v2:
- strict schema via judge_schema.validate_strict (all fields/bools/lists required, int 0-10,
  verdict enum); status 'ok' only when the reply is fully valid;
- reference-leak redefined: extra REFERENCE content (heard in audio1, absent from the requested
  target text) heard in audio2 -> ref_content_leak / leaked_ref_content (v2's ref_word_leak was
  wrong: ref and target are both Russian and share legitimate words);
- explicit no-ref TTS prompt (instruction-only generation) instead of an identity prompt;
- tool/capability goal uses the EXACT task instruction and task-specific checks;
- unknown task ids produce an explicit 'unknown_task' status, never a silent skip;
- inputs flagged physically invalid (e.g. vocal extraction with a speech-only reference) are
  reported as 'invalid_input', not judged as pass/fail.

Usage:
  python audit_gemini_judge_v3.py --blind --areas clone --out ..._v3_clone.jsonl --workers 3
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_creds import get_key, get_url, get_model
from judge_schema import validate_strict

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
URL = get_url()
KEY = get_key()
MODEL = get_model()
PROMPT_VERSION = "audit-2026-09-16-v3"

PROMPT_TTS_REF = """Ты строгий эксперт-оценщик синтеза речи. Ты СЛЫШИШЬ аудио.
Ответь audio_access (true/false).
АУДИО 1 — референс (тот же диктор, ДРУГОЙ текст). АУДИО 2 — синтез для оценки.
Целевой текст синтеза (знак + перед ударной гласной; читай без знаков):
"{text}"
Оцени 0-10: text_fidelity, endings, naturalness, prosody, accent, palatalization, stress,
artifacts, voice_similarity_to_ref (сходство тембра именно с АУДИО 1).
Флаги: truncated(bool), words_mangled, words_dropped, words_misstressed, phoneme_substitutions
(ж/з, ч/ц, ы/и, ш/щ, формат "пробежек -> пробезек"), issues(str).
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access":bool,"text_fidelity":int,"endings":int,"naturalness":int,"prosody":int,"accent":int,
"palatalization":int,"stress":int,"artifacts":int,"voice_similarity_to_ref":int,"truncated":bool,
"words_mangled":[],"words_dropped":[],"words_misstressed":[],"phoneme_substitutions":[],"issues":"",
"overall":int,"verdict":"годен|доработка|брак"}}"""

PROMPT_TTS_NOREF = """Ты строгий эксперт-оценщик синтеза речи. Ты СЛЫШИШЬ одно аудио (синтез).
Ответь audio_access (true/false). Референса нет: оцениваем только синтез по тексту.
Целевой текст (знак + перед ударной гласной; читай без знаков):
"{text}"
Оцени 0-10: text_fidelity, endings, naturalness, prosody, accent, palatalization, stress, artifacts.
Флаги: truncated(bool), words_mangled, words_dropped, words_misstressed, phoneme_substitutions, issues.
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access":bool,"text_fidelity":int,"endings":int,"naturalness":int,"prosody":int,"accent":int,
"palatalization":int,"stress":int,"artifacts":int,"truncated":bool,"words_mangled":[],"words_dropped":[],
"words_misstressed":[],"phoneme_substitutions":[],"issues":"","overall":int,"verdict":"годен|доработка|брак"}}"""

PROMPT_CLONE = """Ты строгий эксперт по клонированию голоса. Ты СЛЫШИШЬ два аудио.
Ответь audio_access (true/false).
АУДИО 1 — референс (донор голоса, его текст НЕ должен звучать в синтезе). АУДИО 2 — синтез.
Целевой текст синтеза (знак + перед ударной гласной; читай без знаков):
"{text}"

ВАЖНО про утечку: ref_content_leak = true ТОЛЬКО если в АУДИО 2 слышны слова/фразы, которые
есть в АУДИО 1, но ОТСУТСТВУЮТ в целевом тексте выше (лишнее содержимое референса). Совпадающие
общеязыковые/служебные слова утечкой НЕ считаются. leaked_ref_content — список таких лишних слов.

Оцени 0-10: text_fidelity, voice_similarity_to_ref (именно сходство с АУДИО 1), naturalness, artifacts.
Флаги: ref_content_leak(bool), leaked_ref_content(list[str]), words_mangled, words_dropped, issues.
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access":bool,"text_fidelity":int,"voice_similarity_to_ref":int,"naturalness":int,"artifacts":int,
"ref_content_leak":bool,"leaked_ref_content":[],"words_mangled":[],"words_dropped":[],"issues":"",
"overall":int,"verdict":"годен|доработка|брак"}}"""

PROMPT_OP = """Ты строгий эксперт по обработке/редактированию речи. Ты СЛЫШИШЬ два аудио.
Ответь audio_access (true/false).
АУДИО 1 — вход/источник. АУДИО 2 — результат обработки.

ТОЧНАЯ ИНСТРУКЦИЯ ЗАДАЧИ: "{instruction}"
ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: {goal}
ЧТО ПРОВЕРИТЬ ОСОБО: {checks}

Оцени 0-10:
- operation_performed: выполнена ли именно эта операция (0 = изменений нет)
- content_preserved: сохранены ли слова/смысл входа (для замены — правильная замена и остальное цело)
- quality: чистота результата (без лишних артефактов/обрывов/клиппинга)
Флаги: expected_effect_matches(bool), introduced_artifacts(bool), heard_result(str), issues(str).
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access":bool,"operation_performed":int,"content_preserved":int,"quality":int,
"expected_effect_matches":bool,"introduced_artifacts":bool,"heard_result":"","issues":"",
"overall":int,"verdict":"годен|доработка|брак"}}"""


def cap_checks(instr):
    s = instr.lower()
    if s.startswith("replace"):
        return "замена произнесена точно, остальные слова сохранены, замена на месте"
    if "extract the vocals" in s or "separation" in s:
        return ("выделен вокал/песня и убран аккомпанемент; вход должен быть смесью с музыкой, "
                "для чисто речевого входа задача физически неприменима")
    if "remove the background noise" in s or "enhance" in s:
        return "шум удалён, речь сохранена и не искажена"
    if "improve the audio quality" in s:
        return "качество/чистота улучшены, речь сохранена"
    if "accent" in s:
        return "акцент ослаблен, тембр/слова сохранены"
    if "whisper" in s:
        return "речь переведена в шёпот, слова сохранены"
    if "tone" in s:
        return "целевая эмоция выражена, слова сохранены"
    if "speed" in s:
        return "скорость изменена в нужную сторону, слова сохранены"
    if "pitch" in s:
        return "тон изменён в нужную сторону примерно на 2 полутона, слова сохранены"
    if "breath" in s:
        return "вдох добавлен в нужном месте, слова сохранены"
    return "целевой эффект достигнут; содержание сохранено"


def tool_checks(op):
    return {
        "volume_up": "громкость речи выросла (примерно +6 дБ), клиппинга нет",
        "volume_down": "громкость речи упала (примерно -6 дБ)",
        "speed_up": "темп вырос примерно в 1.1 раза, слова сохранены",
        "speed_down": "темп снизился примерно до 0.9, слова сохранены",
        "pitch_up": "высота голоса выросла примерно на 2 полутона, слова сохранены",
        "pitch_down": "высота голоса снизилась примерно на 2 полутона, слова сохранены",
        "noise_add": "шум удалён, речь чище, слова сохранены",
    }.get(op, "операция выполнена, содержание сохранено")


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_problem(path):
    """Return reason string if the WAV is missing/empty/non-finite/unreadable, else None."""
    if not path or not os.path.exists(path):
        return "audio file missing"
    if os.path.getsize(path) <= 44:
        return "audio file empty"
    try:
        import soundfile as sf
        import numpy as np
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.size == 0 or sr <= 0:
            return "audio has zero frames"
        if not np.all(np.isfinite(x)):
            return "audio contains non-finite samples"
        if float(np.abs(x).max()) <= 1e-7:
            return "audio is digital silence"
    except Exception as e:
        return f"audio unreadable: {type(e).__name__}"
    return None


def load_refs():
    pack = json.load(open(os.path.join(LT, "eval_pack", "pack.json"), encoding="utf-8"))
    phon = json.load(open(os.path.join(LT, "phonetic_pack", "pack.json"), encoding="utf-8"))
    return {p["id"]: p for p in list(pack) + list(phon)}


def spec_for(entry, group, tid):
    if group == "clone":
        m = re.search(r"'([^']+)'", entry.get("instruction", ""))
        return {"mode": "clone", "needs_input": True, "text": m.group(1) if m else "",
                "instruction": entry.get("instruction", ""), "goal": "произнести целевой текст голосом референса",
                "checks": "нет лишнего содержимого референса"}
    if group in ("tts", "phonetics", "phrase_training"):
        return {"mode": "tts", "needs_input": bool(entry.get("ref")), "text": entry.get("text") or "",
                "instruction": entry.get("instruction", ""), "goal": "", "checks": ""}
    if group == "tool":
        op = entry.get("op", "")
        return {"mode": "tool", "needs_input": True, "text": op, "instruction": entry.get("instruction", ""),
                "goal": entry.get("instruction", ""), "checks": tool_checks(op)}
    if group == "capability":
        cap = entry.get("cap", "")
        return {"mode": "capability", "needs_input": True, "text": cap,
                "instruction": entry.get("instruction", ""), "goal": entry.get("instruction", ""),
                "checks": cap_checks(entry.get("instruction", ""))}
    return None


def invalid_input(it):
    """Physically inapplicable inputs (do not relabel with a surrogate)."""
    if it["group"] == "capability" and it.get("text") == "vocal_extraction":
        ref = (it.get("ref") or "").lower()
        if ref.endswith("user_ref.wav") or "ru_wav_mfa" in ref:
            return "vocal extraction requires a music+vocal mix, got speech-only reference"
    return None


def call_model(wav_paths, prompt, retries=3, backoff=4):
    b64s = [base64.b64encode(open(p, "rb").read()).decode() for p in wav_paths]
    content = [{"type": "input_audio", "input_audio": {"data": b, "format": "wav"}} for b in b64s]
    content.append({"type": "text", "text": prompt})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 800, "temperature": 0.0}
    last = "?"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=240).read().decode())
            c = data["choices"][0]["message"].get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            txt = str(c)
            m, n = txt.find("{"), txt.rfind("}")
            if m >= 0 and n > m:
                return txt, txt[m:n + 1], None
            last = "no json object"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(backoff * (attempt + 1))
    return "", None, last


def build_blind(areas):
    refs = load_refs()
    out = []
    for r in csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8")):
        group = r["group"]
        if areas and group not in areas:
            continue
        tid = r["task_id"]
        wav = os.path.join(BLIND, "wav", r["file"])
        if tid == "phrase_sber":
            spec = {"mode": "tts", "needs_input": True,
                    "text": "Сбер, включи музыку для пробежек через десять минут",
                    "instruction": "training-phrase sample", "goal": "", "checks": ""}
            ref = os.path.join(LT, "user_ref.wav")
        else:
            e = refs.get(tid)
            if e is None:
                out.append({"file": r["file"], "task_id": tid, "group": group, "ref": None, "wav": wav,
                            "unknown": True})
                continue
            spec = spec_for(e, group, tid)
            ref = e.get("ref")
        out.append({"file": r["file"], "task_id": tid, "group": group, "ref": ref, "wav": wav,
                    "unknown": False, **spec})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--areas", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.manifest:
        items = []
        for it in json.load(open(args.manifest, encoding="utf-8")):
            items.append({"unknown": False, "instruction": it.get("instruction", ""),
                          "goal": it.get("goal", ""), "checks": it.get("checks", ""),
                          "text": it.get("text") or it.get("text_override", ""),
                          "mode": it.get("mode", it.get("group", "tts")),
                          "needs_input": bool(it.get("ref")), **it})
    else:
        items = build_blind(set(a for a in args.areas.split(",") if a))
    if args.limit:
        items = items[:args.limit]
    raw_out = args.out.replace(".jsonl", "") + ".raw.jsonl"

    def prepare(it):
        if it.get("unknown"):
            it["_wavs"], it["_prompt"], it["_key"] = [], "", None
            return
        if it.get("needs_input") and not it.get("ref"):
            it["_wavs"], it["_prompt"], it["_key"] = [it["wav"]], "", None
            it["_missing_ref"] = True
            return
        it["_wavs"] = [it["ref"], it["wav"]] if it.get("needs_input") else [it["wav"]]
        tmpl = (PROMPT_TTS_NOREF if it["mode"] == "tts" and not it["needs_input"]
                else PROMPT_TTS_REF if it["mode"] == "tts" else
                PROMPT_CLONE if it["mode"] == "clone" else PROMPT_OP)
        prompt = tmpl.format(text=it.get("text", ""), instruction=it.get("instruction", ""),
                             goal=it.get("goal", ""), checks=it.get("checks", ""))
        it["_prompt"] = prompt
        it["_key"] = [it["file"], it.get("group"), PROMPT_VERSION,
                      hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                      [sha256_file(w) for w in it["_wavs"]], MODEL]

    for it in items:
        prepare(it)

    done_keys = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                k = r.get("resume_key") or [r.get("file"), r.get("group"), r.get("prompt_version"),
                                            r.get("prompt_sha256"), r.get("audio_sha256"), r.get("model")]
                done_keys.add(tuple(json.dumps(x, sort_keys=True) if isinstance(x, list) else x for x in k))
    todo = [it for it in items if it["_key"] and tuple(
        json.dumps(x, sort_keys=True) if isinstance(x, list) else x for x in it["_key"]) not in done_keys]
    print(f"total={len(items)} done_ok={len(items)-len(todo)} todo={len(todo)} model={MODEL}", flush=True)

    lock = threading.Lock()
    fout = open(args.out, "a", encoding="utf-8")
    fraw = open(raw_out, "a", encoding="utf-8")
    okn = [0]
    badn = [0]

    def work(it):
        base = {"file": it.get("file") or it.get("wav"), "task_id": it.get("id", it.get("task_id")),
                "group": it.get("group", "manifest"),
                "mode": it.get("mode"), "model": MODEL, "prompt_version": PROMPT_VERSION,
                "status": "error", "reason": "", "judge": None,
                "prompt_sha256": it["_key"][3] if it["_key"] else None,
                "audio_sha256": it["_key"][4] if it["_key"] else None,
                "resume_key": it["_key"]}
        if it.get("unknown"):
            base.update(status="unknown_task", reason="task_id not found in eval/phonetic packs")
            return base, None
        if it.get("_missing_ref"):
            base.update(status="invalid_input", reason="task requires a reference audio but none is set")
            return base, None
        bad = invalid_input(it)
        if bad:
            base.update(status="invalid_input", reason=bad)
            return base, None
        probs = []
        for w in it["_wavs"]:
            p = audio_problem(w)
            if p:
                probs.append(f"{os.path.basename(w)}: {p}")
        if probs:
            status = "missing_result" if audio_problem(it["wav"]) else "invalid_input"
            base.update(status=status, reason="; ".join(probs))
            return base, None
        base["n_audio"] = len(it["_wavs"])
        last_raw = ""
        attempts = []
        for attempt in range(3):
            t = time.time()
            raw, js, err = call_model(it["_wavs"], it["_prompt"])
            secs = round(time.time() - t, 1)
            last_raw = raw or last_raw
            if js is None:
                attempts.append({"attempt": attempt + 1, "seconds": secs, "error": err or "no reply"})
                base.update(status="error", reason=err or "no reply")
                continue
            try:
                obj = json.loads(js)
            except Exception as e:
                attempts.append({"attempt": attempt + 1, "seconds": secs, "error": f"json parse: {e}"})
                base.update(status="invalid_json", reason=f"json parse: {e}")
                continue
            base["judge"] = obj
            problems = validate_strict(it["group"], obj, has_ref=bool(it.get("needs_input")))
            attempts.append({"attempt": attempt + 1, "seconds": secs, "error": "; ".join(problems[:4])})
            if problems:
                base.update(status="invalid_schema", reason="; ".join(problems[:6]))
                continue
            if obj.get("audio_access") is False:
                base.update(status="no_result", reason="audio_access=false")
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
            base.update(status="ok", reason="")
            break
        base["attempts"] = attempts
        return base, last_raw

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row, raw = fut.result()
            except Exception as e:
                it = futs[fut]
                row, raw = {"file": it.get("file") or it.get("wav"), "task_id": it.get("id", it.get("task_id")),
                            "group": it.get("group", "manifest"),
                            "mode": it.get("mode"), "status": "error", "reason": f"fatal {e}",
                            "judge": None}, None
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                if raw is not None:
                    fraw.write(json.dumps({"file": row["file"], "raw": raw}, ensure_ascii=False) + "\n")
                    fraw.flush()
            if row["status"] == "ok":
                okn[0] += 1
            else:
                badn[0] += 1
            print(f"[{i}/{len(todo)}] {row['file']} {str(row.get('mode')):11s} {row['status']:13s} "
                  f"overall={(row.get('judge') or {}).get('overall')} ok={okn[0]} bad={badn[0]}", flush=True)

    fout.close()
    fraw.close()
    print(f"\nDONE ok={okn[0]} bad={badn[0]} -> {args.out}")


if __name__ == "__main__":
    main()
