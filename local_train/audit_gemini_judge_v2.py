"""Gemini audit judge v2 (16.09.2026).

Fixes over v1:
- task-specific expected-result rubrics (tool op / capability), not one generic content_preserved;
- explicit audio routing: ref is sent only when the task genuinely needs an input audio;
  missing required ref -> no_result (never a fake two-audio call);
- voice_similarity_to_ref compares to the reference, not "naturalness";
- JSON schema validation (invalid JSON != success), audio_access required;
- full raw replies kept in a separate JSONL, SHA256 of every audio, model alias,
  prompt_version + prompt_sha256, response seconds;
- retries only on network/parse errors; resume skips only rows with status=ok and valid schema.

Usage:
  python audit_gemini_judge_v2.py --manifest preflight_manifest.json --out results.jsonl --workers 3
  python audit_gemini_judge_v2.py --blind --areas tts,phonetics --workers 3
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

from gemini_creds import get_key, get_model, get_url

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
URL = get_url()
KEY = get_key()
MODEL_DEFAULT = get_model()
PROMPT_VERSION = "audit-2026-09-16-v2"

SCORE_KEYS = ["overall"]
TTS_KEYS = ["text_fidelity", "endings", "naturalness", "prosody", "accent", "palatalization",
            "stress", "artifacts", "voice_similarity_to_ref"]
CLONE_KEYS = ["text_fidelity", "voice_similarity_to_ref", "naturalness", "artifacts"]
OP_KEYS = ["operation_performed", "content_preserved", "quality"]

TOOL_GOALS = {
    "volume_up": "Сделать речь ГРОМЧЕ примерно на 6 дБ, содержание не менять.",
    "volume_down": "Сделать речь ТИШЕ примерно на 6 дБ, содержание не менять.",
    "speed_up": "УСКОРИТЬ речь примерно в 1.1 раза, содержание не менять.",
    "speed_down": "ЗАМЕДЛИТЬ речь примерно до 0.9 скорости, содержание не менять.",
    "pitch_up": "Поднять ТОН голоса примерно на 2 полутона, содержание не менять.",
    "pitch_down": "Понизить ТОН голоса примерно на 2 полутона, содержание не менять.",
    "noise_add": "УБРАТЬ фоновый шум, сделать голос чище, содержание не менять.",
}
CAP_GOALS = {
    "emotion_happy": "Произнести вход РАДОСТНО (happy), сохранив слова.",
    "emotion_sad": "Произнести вход ГРУСТНО (sad), сохранив слова.",
    "emotion_angry": "Произнести вход ЗЛОБНО (angry), сохранив слова.",
    "whisper": "Превратить вход в ШЁПОТ, сохранив слова.",
    "speech_edit": "Заменить слово 'привет' на 'здравствуйте', остальное сохранить.",
    "deaccent": "УБРАТЬ акцент, сохранив тот же голос и слова.",
    "enhancement": "УБРАТЬ фоновый шум, сделать голос чище, содержание сохранить.",
    "quality": "УЛУЧШИТЬ качество звука, сделать чище, содержание сохранить.",
    "speed_12": "Изменить скорость речи до 1.2x, содержание сохранить.",
    "pitch_up": "Поднять тон примерно на 2 полутона, содержание сохранить.",
    "nonverbal_breath": "ДОБАВИТЬ вдох перед последним словом, слова сохранить.",
    "vocal_extraction": "ВЫДЕЛИТЬ вокал и убрать аккомпанемент.",
}

PROMPT_TTS = """Ты строгий эксперт-оценщик синтеза речи. Ты СЛЫШИШЬ аудио (нативная аудиомодальность).
Сначала ответь, реально ли доступны аудио: поле audio_access (true/false).

АУДИО 1 — референс (тот же диктор, ДРУГОЙ текст). АУДИО 2 — синтез, который оцениваем.
Целевой текст синтеза (знак + перед ударной гласной; читай без знаков):
"{text}"

Оцени ЦЕЛОЧИСЛЕННО 0-10:
- text_fidelity: все слова/слоги на месте (10 = полностью; 3 = выпал слог внутри слова)
- endings: конечные согласные/слоги произнесены
- naturalness: живость, не робот
- prosody: ритм и интонация
- accent: 10 = native; 3 = явный акцент
- palatalization: мягкие/твёрдые согласные
- stress: ударения
- artifacts: чистота (10 = нет шума; 0 = треск/обрывы)
- voice_similarity_to_ref: насколько тембр синтеза похож ИМЕННО на АУДИО 1 (10 = тот же человек)
Флаги: words_mangled ("правильно -> услышано"), words_dropped, words_misstressed,
phoneme_substitutions (ж/з, ч/ц, ы/и, ш/щ), truncated (bool), issues (1-2 предложения).
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access": bool, "text_fidelity": int, "endings": int, "naturalness": int, "prosody": int,
"accent": int, "palatalization": int, "stress": int, "artifacts": int, "voice_similarity_to_ref": int,
"truncated": bool, "words_mangled": [], "words_dropped": [], "words_misstressed": [],
"phoneme_substitutions": [], "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

PROMPT_CLONE = """Ты строгий эксперт по клонированию голоса. Ты СЛЫШИШЬ два аудио.
Сначала ответь audio_access (true/false).
АУДИО 1 — референс (донор голоса). АУДИО 2 — синтез.
Целевой текст синтеза (знак + перед ударной гласной; читай без знаков):
"{text}"

Оцени 0-10:
- text_fidelity: текст произнесён полностью и точно
- voice_similarity_to_ref: тембр синтеза похож ИМЕННО на АУДИО 1 (10 = тот же человек)
- naturalness: живость
- artifacts: чистота
- ref_word_leak: true, если во ВТОРОМ слышны слова из ПЕРВОГО (утечка референса)
Флаги: leaked_ref_words, words_mangled, words_dropped, issues.
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access": bool, "text_fidelity": int, "voice_similarity_to_ref": int, "naturalness": int,
"artifacts": int, "ref_word_leak": bool, "leaked_ref_words": [], "words_mangled": [],
"words_dropped": [], "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

PROMPT_OP = """Ты строгий эксперт по обработке/редактированию речи. Ты СЛЫШИШЬ два аудио.
Сначала ответь audio_access (true/false).
АУДИО 1 — вход/источник. АУДИО 2 — результат обработки.

ЗАДАЧА: {task_name}
ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: {goal}
ЧТО ПРОВЕРИТЬ ОСОБО: {checks}

Оцени 0-10:
- operation_performed: насколько выполнена именно эта операция (10 = явно и корректно; 0 = нет изменений)
- expected_effect_matches: bool, соответствует ли результат ожидаемому результату выше
- content_preserved: сохранены ли слова/смысл входа (для замены — правильная замена и остальное цело)
- quality: чистота результата (без лишних артефактов/обрывов/клиппинга)
- introduced_artifacts: bool, появились ли новые дефекты (вставки, повторы, треск, обрыв)
Флаги: heard_result (что реально услышано), issues.
Ответь ТОЛЬКО JSON без markdown:
{{"audio_access": bool, "operation_performed": int, "expected_effect_matches": bool,
"content_preserved": int, "quality": int, "introduced_artifacts": bool, "heard_result": "string",
"issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

BOOL_KEYS = {"truncated", "ref_word_leak", "expected_effect_matches", "introduced_artifacts", "audio_access"}
LIST_KEYS = {"words_mangled", "words_dropped", "words_misstressed", "phoneme_substitutions",
             "leaked_ref_words"}
STR_KEYS = {"issues", "verdict", "heard_result"}


def prompts_for(mode):
    if mode in ("tts", "phonetics", "phrase_training"):
        return PROMPT_TTS, TTS_KEYS
    if mode == "clone":
        return PROMPT_CLONE, CLONE_KEYS
    return PROMPT_OP, OP_KEYS


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(mode, obj):
    problems = []
    if not isinstance(obj, dict):
        return ["not an object"]
    if "audio_access" not in obj:
        problems.append("missing audio_access")
    required = {"audio_access": bool, "overall": int, "verdict": str}
    for k in required:
        if k not in obj:
            problems.append("missing " + k)
    _, score_keys = prompts_for(mode)
    for k in score_keys + ["overall"]:
        v = obj.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            problems.append(f"{k} not numeric")
        elif not (0 <= v <= 10):
            problems.append(f"{k} out of range")
    for k in BOOL_KEYS & set(obj):
        if not isinstance(obj[k], bool):
            problems.append(f"{k} not bool")
    for k in LIST_KEYS & set(obj):
        if not isinstance(obj[k], list):
            problems.append(f"{k} not list")
    for k, v in obj.items():
        if k in STR_KEYS and not isinstance(v, str):
            problems.append(f"{k} not string")
    if obj.get("verdict") not in ("годен", "доработка", "брак", None):
        problems.append("bad verdict")
    return problems


def call_model(model, wav_paths, prompt, retries=3, backoff=4):
    b64s = [base64.b64encode(open(p, "rb").read()).decode() for p in wav_paths]
    content = [{"type": "input_audio", "input_audio": {"data": b, "format": "wav"}} for b in b64s]
    content.append({"type": "text", "text": prompt})
    body = {"model": model, "messages": [{"role": "user", "content": content}],
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
            last = "no json object in reply"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(backoff * (attempt + 1))
    return "", None, last


def load_refs():
    pack = json.load(open(os.path.join(LT, "eval_pack", "pack.json"), encoding="utf-8"))
    phon = json.load(open(os.path.join(LT, "phonetic_pack", "pack.json"), encoding="utf-8"))
    return {p["id"]: p for p in list(pack) + list(phon)}


def task_spec(entry, group, task_id):
    if group == "clone":
        m = re.search(r"'([^']+)'", entry.get("instruction", ""))
        return {"mode": "clone", "needs_input": True, "text": m.group(1) if m else "",
                "task_name": "клонирование голоса", "goal": "произнести целевой текст голосом референса",
                "checks": "нет утечки слов референса"}
    if group in ("tts", "phonetics", "phrase_training"):
        text = entry.get("text") if entry else None
        return {"mode": group if group != "phrase_training" else "tts", "needs_input": True,
                "text": text or "Сбер, включи музыку для пробежек через десять минут",
                "task_name": "синтез речи", "goal": "", "checks": ""}
    if group == "tool":
        op = entry.get("op", "")
        return {"mode": "tool", "needs_input": True, "text": op,
                "task_name": f"инструмент: {op}", "goal": TOOL_GOALS.get(op, entry.get("instruction", "")),
                "checks": "направление и порядок величины; содержание не потеряно"}
    if group == "capability":
        cap = entry.get("cap", "")
        return {"mode": "capability", "needs_input": True, "text": cap,
                "task_name": f"capability: {cap}", "goal": CAP_GOALS.get(cap, entry.get("instruction", "")),
                "checks": "целевой эффект достигнут; содержание сохранено"}
    return {"mode": group, "needs_input": True, "text": "", "task_name": group, "goal": "", "checks": ""}


def blind_items(areas):
    refs = load_refs()
    rows = list(csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        group = r["group"]
        if areas and group not in areas:
            continue
        tid = r["task_id"]
        entry = refs.get(tid)
        if tid == "phrase_sber":
            spec = {"mode": "tts", "needs_input": True,
                    "text": "Сбер, включи музыку для пробежек через десять минут",
                    "task_name": "синтез фразы обучения", "goal": "", "checks": ""}
            ref = os.path.join(LT, "user_ref.wav")
        elif entry:
            spec = task_spec(entry, group, tid)
            ref = entry.get("ref")
        else:
            continue
        out.append({"file": r["file"], "task_id": tid, "group": group, "ref": ref,
                    "wav": os.path.join(BLIND, "wav", r["file"]), **spec})
    return out


def manifest_items(path):
    man = json.load(open(path, encoding="utf-8"))
    out = []
    for it in man:
        spec = it.get("spec") or {}
        merged = dict(it)
        merged.update(spec)
        merged.setdefault("mode", it.get("group", "tts"))
        merged.setdefault("needs_input", bool(it.get("ref")))
        merged.setdefault("task_name", it.get("task_id", ""))
        merged.setdefault("goal", "")
        merged.setdefault("checks", "")
        if not merged.get("text"):
            merged["text"] = it.get("text_override", "")
        out.append(merged)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--areas", default="")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out")
    args = ap.parse_args()

    items = manifest_items(args.manifest) if args.manifest else blind_items(set(a for a in args.areas.split(",") if a))
    if args.limit:
        items = items[:args.limit]
    raw_out = args.raw_out or (args.out.replace(".jsonl", "") + ".raw.jsonl")

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok" and not validate(row.get("mode", "tts"), row.get("judge") or {}):
                done[row["file"]] = row
    todo = [it for it in items if it["file"] not in done]
    print(f"total={len(items)} done_ok={len(done)} todo={len(todo)} workers={args.workers} "
          f"model={args.model}", flush=True)

    lock = threading.Lock()
    fout = open(args.out, "a", encoding="utf-8")
    fraw = open(raw_out, "a", encoding="utf-8")
    okn = [0]
    errn = [0]

    def work(it):
        prompt_tmpl, _ = prompts_for(it["mode"])
        prompt = prompt_tmpl.format(text=it["text"], task_name=it.get("task_name", ""),
                                    goal=it.get("goal", ""), checks=it.get("checks", ""))
        if it["needs_input"] and (not it.get("ref") or not os.path.exists(it["ref"])):
            return {"file": it["file"], "task_id": it["task_id"], "group": it["group"],
                    "mode": it["mode"], "status": "no_result", "reason": "missing required input ref",
                    "judge": None}, None
        wavs = ([it["ref"], it["wav"]] if it["needs_input"] else [it["wav"]])
        wavs = [w for w in wavs if w and os.path.exists(w)]
        row = {"file": it["file"], "task_id": it["task_id"], "group": it["group"], "mode": it["mode"],
               "model": args.model, "prompt_version": PROMPT_VERSION,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
               "audio_sha256": [sha256_file(w) for w in wavs], "n_audio": len(wavs),
               "seconds": 0, "judge": None, "status": "error", "reason": "no attempt"}
        last_raw = ""
        for attempt in range(3):
            t = time.time()
            raw, js, err = call_model(args.model, wavs, prompt)
            row["seconds"] = round(time.time() - t, 1)
            last_raw = raw or last_raw
            if js is None:
                row.update(status="error", reason=err or "no reply", judge=None)
                continue
            try:
                obj = json.loads(js)
            except Exception as e:
                row.update(status="invalid_json", reason=f"json parse: {e}", judge=None)
                continue
            probs = validate(it["mode"], obj)
            row["judge"] = obj
            if probs:
                row.update(status="invalid_schema", reason="; ".join(probs))
                continue
            if obj.get("audio_access") is False:
                row.update(status="no_result", reason="audio_access=false")
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
            row.update(status="ok", reason="")
            break
        return row, last_raw

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row, raw = fut.result()
            except Exception as e:
                it = futs[fut]
                row, raw = {"file": it["file"], "task_id": it["task_id"], "group": it["group"],
                            "mode": it["mode"], "status": "error", "reason": f"fatal {e}",
                            "judge": None}, None
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                if raw is not None:
                    fraw.write(json.dumps({"file": row["file"], "raw": raw},
                                          ensure_ascii=False) + "\n")
                    fraw.flush()
            if row["status"] == "ok":
                okn[0] += 1
            else:
                errn[0] += 1
            print(f"[{i}/{len(todo)}] {row['file']} {row['mode']:11s} {row['status']:13s} "
                  f"overall={(row.get('judge') or {}).get('overall')} ({row.get('seconds')}s, "
                  f"ok={okn[0]} bad={errn[0]})", flush=True)

    fout.close()
    fraw.close()
    print(f"\nDONE ok={okn[0]} bad={errn[0]} -> {args.out}\nraw -> {raw_out}")


if __name__ == "__main__":
    main()
