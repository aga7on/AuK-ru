"""Detailed Russian TTS evaluation via OpenRouter (free NVIDIA Omni reasoning model).

Usage:
  python openrouter_judge.py --gen-dir G:\\AI\\AuK\\local_tests\\variations_v3 --limit 4
  python openrouter_judge.py --audio path.wav --text "..." [--ref path.wav]

Key resolution: env OPENROUTER_API_KEY or file G:\\AI\\_tmp\\openrouter_key.txt (first line).
"""
import argparse
import base64
import json
import os
import re
import sys
import time

import requests

API = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = r"G:\AI\_tmp\openrouter_key.txt"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

PROMPT = """Мы оцениваем СИНТЕЗ РУССКОЙ РЕЧИ (важно: язык именно русский, не английский).
SAMPLE — сгенерированная моделью русская речь. REFERENCE — эталонный голос того же диктора (если приложен).
Целевой текст ниже. Метки: «+» стоит перед ударной гласной, апостроф «'» обозначает мягкость/твёрдость; при чтении разметку игнорируй.

ЦЕЛЕВОЙ ТЕКСТ: "{text}"

Оцени СТРОГО по критериям, каждый — целое 0–10:
- text_fidelity: все слова и слоги на месте (пропуск слога внутри слова — 3 и ниже);
- endings: договорённость окончаний (слова проговорены до конца, конечные согласные слышны);
- naturalness: живость, не «робот»;
- prosody: ритм и интонация (нет лишних пауз, плавная мелодика, единый темп);
- accent: 10 = носитель русского; 0 = сильный иностранный акцент;
- palatalization: мягкость ь/ъ (день, семья, объём, съел);
- stress: ударения верны (перечисли ошибочные в words_misstressed);
- artifacts: чистота (шум/песочность/щелчки);
- voice_similarity: сходство с REFERENCE (если референса нет — null).

Флаги: same_gender (true/false), text_matches (true/false), truncated (true/false),
native_ok (true/false), words_dropped (["..."]), words_mangled ([{"correct": "...", "heard": "..."}]),
words_misstressed (["..."]), issues (1–2 предложения по-русски), overall (0–10), verdict ("годен" | "доработка" | "брак").

Ответ — ТОЛЬКО JSON-объект без markdown и текста вокруг."""


def read_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    if os.path.isfile(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.readline().strip()
    return ""


def b64_wav(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def call_openrouter(key, model, content, timeout=240, retries=2):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://linzhanmou.com/unimate/", "X-Title": "AuK-RU-eval"}
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0.2, "max_tokens": 1200}
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(API, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(5 + 10 * attempt)
    raise RuntimeError(last)


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"_raw": text, "_parse_error": True}
    return {"_raw": text, "_parse_error": True}


def judge_one(key, model, gen_path, text, ref_path=None):
    content = [{"type": "text", "text": PROMPT.replace("{text}", text)},
               {"type": "input_audio", "input_audio": {"data": b64_wav(gen_path), "format": "wav"}}]
    if ref_path and os.path.isfile(ref_path):
        content.append({"type": "input_audio", "input_audio": {"data": b64_wav(ref_path), "format": "wav"}})
    try:
        reply = call_openrouter(key, model, content)
    except RuntimeError as exc:
        if ref_path and "400" in str(exc):
            content = content[:2]
            reply = call_openrouter(key, model, content)
        else:
            raise
    return parse_json(reply), reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir")
    ap.add_argument("--audio")
    ap.add_argument("--text")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = read_key()
    if not key:
        print("NO_KEY: положи OpenRouter API key в G:\\AI\\_tmp\\openrouter_key.txt или env OPENROUTER_API_KEY")
        sys.exit(2)

    items = []
    if args.gen_dir:
        man = json.load(open(os.path.join(args.gen_dir, "manifest.json"), encoding="utf-8"))
        for m in man:
            if not os.path.isfile(m.get("gen", "")):
                continue
            items.append({"gen": m["gen"], "text": m.get("text", ""), "ref": m.get("ref")})
    elif args.audio and args.text:
        items.append({"gen": args.audio, "text": args.text, "ref": args.ref})
    else:
        print("need --gen-dir or --audio+--text")
        sys.exit(2)
    if args.limit:
        items = items[: args.limit]

    results = []
    for it in items:
        t0 = time.time()
        try:
            judge, raw = judge_one(key, args.model, it["gen"], it["text"], it.get("ref"))
            rec = {"gen": it["gen"], "text": it["text"], "ref": it.get("ref"),
                   "judge": judge, "raw": raw[:2000], "sec": round(time.time() - t0, 1)}
            results.append(rec)
            j = judge
            print("[%s] ovrl=%s verdict=%s text=%s endings=%s prosody=%s stress=%s dropped=%s issues=%s (%.0fs)" % (
                os.path.basename(it["gen"]), j.get("overall"), j.get("verdict"), j.get("text_fidelity"),
                j.get("endings"), j.get("prosody"), j.get("stress"), str(j.get("words_dropped"))[:30],
                (j.get("issues") or "")[:70], rec["sec"]), flush=True)
        except Exception as exc:
            print(f"[FAIL] {os.path.basename(it['gen'])}: {exc}", flush=True)
            results.append({"gen": it["gen"], "error": str(exc)})

    out = args.out or (os.path.join(args.gen_dir, "results_openrouter.json") if args.gen_dir else
                       os.path.splitext(args.audio)[0] + ".openrouter.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved:", out)


if __name__ == "__main__":
    main()
