"""Blind Gemini audit judge for the s2 blind set (variants hidden, resumable, concurrent).

Builds its task list from local_tests/blind_s2/LISTEN.csv + the two eval packs, sends
WAV audio (base64, input_audio) to the local Gemini proxy and writes JSONL results
incrementally (safe to kill and restart: finished files are skipped).

Usage:
  python audit_gemini_judge.py --areas tts,phonetics,phrase_training,capability,tool,clone
  python audit_gemini_judge.py --areas phonetics --workers 4
"""
import argparse
import base64
import csv
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_creds import get_key, get_model, get_url

KEY = get_key()
URL = get_url()
AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
PROMPT_VERSION = "audit-2026-09-16-v1"

PROMPT_TTS = """Ты строгий эксперт-оценщик синтезированной русской речи. Ты СЛЫШИШЬ аудио (нативная аудио-модальность).

Целевой текст (знак + перед ударной гласной; читай без знаков):
"{text}"

Оцени ЦЕЛОЧИСЛЕННО 0-10:
- text_fidelity: все слова и слоги на месте (10 = полностью; 5 = слово проглочено; 3 и ниже = слог внутри слова выпал)
- endings: конечные согласные/слоги произнесены до конца
- naturalness: живость, не робот (10 = неотличим от диктора; 5 = синтетично/монотонно)
- prosody: ритм и интонация
- accent: 10 = native; 7 = лёгкий акцент; 3 = явный акцент
- palatalization: мягкие/твёрдые согласные (день, семья, объём, съел, роль)
- stress: ударения
- artifacts: чистота (10 = нет шума; 0 = треск, обрывы)
- voice_similarity: насколько голос похож на естественного русского диктора

Флаги:
- words_mangled: список слов с искажением (формат "правильно -> услышано")
- words_dropped: список пропущенных слов
- phoneme_substitutions: список замен по типу ж/з, ч/ц, ы/и, ш/щ (формат "пробежек -> пробезек")
- truncated: bool
- issues: 1-2 предложения конкретно по фонеме/слову

Ответь ТОЛЬКО JSON-объектом без markdown, ровно с этими ключами:
{{"text_fidelity": int, "endings": int, "naturalness": int, "prosody": int, "accent": int, "palatalization": int, "stress": int, "artifacts": int, "truncated": bool, "words_mangled": [], "words_dropped": [], "phoneme_substitutions": [], "words_misstressed": [], "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

PROMPT_CLONE = """Ты строгий эксперт-оценщик клонирования голоса. Ты СЛЫШИШЬ ДВА аудио: ПЕРВОЕ — референс (донор голоса), ВТОРОЕ — синтез.

Синтез должен произнести этот текст (знак + перед ударной гласной; читай без знаков):
"{text}"

Оцени ЦЕЛОЧИСЛЕННО 0-10:
- text_fidelity: текст произнесён полностью и точно
- voice_similarity_to_ref: насколько тембр/голос синтеза похож ИМЕННО на ПЕРВОЕ (референс), 10 = тот же человек
- naturalness: живость
- artifacts: чистота
- ref_word_leak: true, если во ВТОРОМ слышны слова/содержимое ПЕРВОГО (утечка содержимого референса)
Флаги: words_mangled, words_dropped, issues.
Ответь ТОЛЬКО JSON без markdown:
{{"text_fidelity": int, "voice_similarity_to_ref": int, "naturalness": int, "artifacts": int, "ref_word_leak": bool, "words_mangled": [], "words_dropped": [], "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

PROMPT_OP = """Ты строгий эксперт по обработке/редактированию речи. Ты СЛЫШИШЬ ДВА аудио: ПЕРВОЕ — вход (источник), ВТОРОЕ — результат обработки.

Требуемая операция/эффект:
"{text}"

Оцени ЦЕЛОЧИСЛЕННО 0-10:
- operation_performed: выполнена ли операция/эффект (10 = явно и корректно; 0 = изменений нет)
- content_preserved: сохранено ли речевое содержание входа (слова, смысл)
- quality: чистота результата (без лишних артефактов, обрывов, клиппинга)
- artifact_of_editing: false, если слышны лишние вставки/повторы/обрывы
Флаги: heard_operation (что реально услышано), issues.
Ответь ТОЛЬКО JSON без markdown:
{{"operation_performed": int, "content_preserved": int, "quality": int, "artifact_of_editing": bool, "heard_operation": "string", "issues": "string", "overall": int, "verdict": "годен|доработка|брак"}}"""

RUBRICS = {
    "tts": PROMPT_TTS,
    "phonetics": PROMPT_TTS,
    "phrase_training": PROMPT_TTS,
    "clone": PROMPT_CLONE,
    "tool": PROMPT_OP,
    "capability": PROMPT_OP,
}
TWO_AUDIO = {"clone", "tool", "capability"}

_lock = threading.Lock()


def load_refs():
    pack = json.load(open(os.path.join(LT, "eval_pack", "pack.json"), encoding="utf-8"))
    phon = json.load(open(os.path.join(LT, "phonetic_pack", "pack.json"), encoding="utf-8"))
    refs = {}
    for p in list(pack) + list(phon):
        refs[p["id"]] = p
    return refs


def task_text(entry, group):
    if group == "clone":
        import re
        m = re.search(r"'([^']+)'", entry.get("instruction", ""))
        return m.group(1) if m else ""
    if group in ("tool", "capability"):
        return entry.get("instruction", "")
    return entry.get("text") or ""


def build_items(areas):
    refs = load_refs()
    rows = list(csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8")))
    items = []
    for r in rows:
        group = r["group"]
        if areas and group not in areas:
            continue
        tid = r["task_id"]
        if tid == "phrase_sber":
            text = "Сбер, включи музыку для пробежек через десять минут"
            ref = os.path.join(LT, "user_ref.wav")
        else:
            e = refs.get(tid)
            if not e:
                continue
            text = task_text(e, group)
            ref = e.get("ref")
        items.append({"file": r["file"], "task_id": tid, "group": group,
                      "text": text, "ref": ref,
                      "wav": os.path.join(BLIND, "wav", r["file"])})
    return items


def call_model(model, wav_paths, text, group, retries=3):
    b64s = [base64.b64encode(open(p, "rb").read()).decode() for p in wav_paths]
    content = []
    for i, b64 in enumerate(b64s):
        content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
    prompt = RUBRICS[group].format(text=text)
    content.append({"type": "text", "text": prompt})
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 700, "temperature": 0.0}
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=240).read().decode())
            c = data["choices"][0]["message"].get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            txt = str(c)
            m = txt.find("{")
            n = txt.rfind("}")
            if m >= 0 and n > m:
                return json.loads(txt[m:n + 1]), ""
            return None, txt[:400]
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(3)
    return None, f"ERROR {last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="tts,phonetics,phrase_training,tool,capability,clone")
    ap.add_argument("--model", default="gemini-3.8-flash-medium")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(LT, "blind_s2", "results_audit_gemini.jsonl"))
    args = ap.parse_args()

    areas = set(a for a in args.areas.split(",") if a)
    items = build_items(areas)
    if args.limit:
        items = items[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    todo = [it for it in items if it["file"] not in done]
    print(f"total={len(items)} done={len(done)} todo={len(todo)} workers={args.workers}", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    n_ok = [0]
    n_err = [0]
    t0 = time.time()

    def work(it):
        wavs = [it["ref"], it["wav"]] if it["group"] in TWO_AUDIO and it["ref"] and os.path.exists(it["ref"]) else [it["wav"]]
        t = time.time()
        obj, raw = call_model(args.model, wavs, it["text"], it["group"])
        row = {"file": it["file"], "task_id": it["task_id"], "group": it["group"],
               "text": it["text"], "model": args.model, "prompt_version": PROMPT_VERSION,
               "n_audio": len(wavs), "seconds": round(time.time() - t, 1),
               "judge": obj, "raw": "" if obj else raw[:600]}
        return row, obj is not None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row, ok = fut.result()
            except Exception as e:
                it = futs[fut]
                row = {"file": it["file"], "task_id": it["task_id"], "group": it["group"],
                       "judge": None, "raw": f"FATAL {e}", "model": args.model,
                       "prompt_version": PROMPT_VERSION}
                ok = False
            with _lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            if ok:
                n_ok[0] += 1
                ov = row["judge"].get("overall")
            else:
                n_err[0] += 1
                ov = "ERR"
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {row['file']} {row['group']:14s} overall={ov} "
                  f"({row.get('seconds')}s, ok={n_ok[0]} err={n_err[0]}, {el:.0f}s)", flush=True)
            if not ok:
                print("    RAW:", str(row.get("raw"))[:200], flush=True)

    out.close()
    print(f"\nJUDGE_DONE ok={n_ok[0]} err={n_err[0]} -> {args.out}")


if __name__ == "__main__":
    main()
