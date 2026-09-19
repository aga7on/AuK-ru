"""Automated audio judge via local Gemini proxy (OpenAI-compatible, native audio input).

Usage:
  python auto_judge.py --dir G:\AI\AuK\local_tests\respell4_ab [--model gemini-3.8-flash-medium] [--limit N]

Reads manifest.json from dir (entries: file + text), sends each wav (base64) with the
review rubric, expects strict JSON, writes results.json incrementally.
"""
import argparse
import base64
import json
import os
import time
import urllib.request

KEY = os.environ.get("AUK_GEMINI_KEY", "")  # sanitized for public release
URL = "http://127.0.0.1:8045/v1/chat/completions"

PROMPT = """РўС‹ СЃС‚СЂРѕРіРёР№ СЌРєСЃРїРµСЂС‚-РѕС†РµРЅС‰РёРє СЃРёРЅС‚РµР·РёСЂРѕРІР°РЅРЅРѕР№ СЂСѓСЃСЃРєРѕР№ СЂРµС‡Рё. РўС‹ РЎР›Р«РЁРРЁР¬ Р°СѓРґРёРѕ (РЅР°С‚РёРІРЅР°СЏ Р°СѓРґРёРѕ-РјРѕРґР°Р»СЊРЅРѕСЃС‚СЊ).

Р¦РµР»РµРІРѕР№ С‚РµРєСЃС‚ (Р·РЅР°Рє + РїРµСЂРµРґ СѓРґР°СЂРЅРѕР№ РіР»Р°СЃРЅРѕР№; С‡РёС‚Р°Р№ Р±РµР· Р·РЅР°РєРѕРІ):
"{text}"

РћС†РµРЅРё Р¦Р•Р›РћР§РРЎР›Р•РќРќРћ 0-10:
- text_fidelity: РІСЃРµ СЃР»РѕРІР° Рё СЃР»РѕРіРё РЅР° РјРµСЃС‚Рµ (10 = РїРѕР»РЅРѕСЃС‚СЊСЋ; 5 = СЃР»РѕРІРѕ РїСЂРѕРіР»РѕС‡РµРЅРѕ; 3 Рё РЅРёР¶Рµ = СЃР»РѕРі РІРЅСѓС‚СЂРё СЃР»РѕРІР° РІС‹РїР°Р», РЅР°РїСЂ. "СЂР°Р·РІРёС‚РёРµ"->"СЂР°Р·РёС‚РёРµ")
- endings: РєРѕРЅРµС‡РЅС‹Рµ СЃРѕРіР»Р°СЃРЅС‹Рµ/СЃР»РѕРіРё РїСЂРѕРёР·РЅРµСЃРµРЅС‹ РґРѕ РєРѕРЅС†Р°
- naturalness: Р¶РёРІРѕСЃС‚СЊ, РЅРµ СЂРѕР±РѕС‚ (10 = РЅРµРѕС‚Р»РёС‡РёРј РѕС‚ РґРёРєС‚РѕСЂР°; 5 = СЃРёРЅС‚РµС‚РёС‡РЅРѕ/РјРѕРЅРѕС‚РѕРЅРЅРѕ)
- prosody: СЂРёС‚Рј Рё РёРЅС‚РѕРЅР°С†РёСЏ (10 = РїР»Р°РІРЅР°СЏ РјРµР»РѕРґРёСЏ, Р»РѕРіРёС‡РЅС‹Рµ РїР°СѓР·С‹; 5 = СЂРІР°РЅР°СЏ/РёСЃРєСѓСЃСЃС‚РІРµРЅРЅР°СЏ)
- accent: 10 = native; 7 = Р»С‘РіРєРёР№ Р°РєС†РµРЅС‚; 3 = СЏРІРЅС‹Р№ Р°РєС†РµРЅС‚
- palatalization: РјСЏРіРєРёРµ/С‚РІС‘СЂРґС‹Рµ СЃРѕРіР»Р°СЃРЅС‹Рµ (РґРµРЅСЊ, СЃРµРјСЊСЏ, РѕР±СЉС‘Рј, СЃСЉРµР», СЂРѕР»СЊ)
- stress: СѓРґР°СЂРµРЅРёСЏ (РїРµСЂРµС‡РёСЃР»Рё РЅРµРїСЂР°РІРёР»СЊРЅС‹Рµ РІ words_misstressed)
- artifacts: С‡РёСЃС‚РѕС‚Р° (10 = РЅРµС‚ С€СѓРјР°; 5 = РїРµСЃРѕРє/С€РёРїРµРЅРёРµ; 0 = С‚СЂРµСЃРє, РѕР±СЂС‹РІС‹)
- voice_similarity: РЅР°СЃРєРѕР»СЊРєРѕ РїРѕС…РѕР¶ РіРѕР»РѕСЃ РЅР° РµСЃС‚РµСЃС‚РІРµРЅРЅРѕРіРѕ СЂСѓСЃСЃРєРѕРіРѕ РґРёРєС‚РѕСЂР°

Р¤Р»Р°РіРё:
- words_mangled: СЃРїРёСЃРѕРє СЃР»РѕРІ, РїСЂРѕРёР·РЅРµСЃС‘РЅРЅС‹С… СЃ РёСЃРєР°Р¶РµРЅРёРµРј (С„РѕСЂРјР°С‚ "РїСЂР°РІРёР»СЊРЅРѕ -> СѓСЃР»С‹С€Р°РЅРѕ")
- words_dropped: СЃРїРёСЃРѕРє РїСЂРѕРїСѓС‰РµРЅРЅС‹С… СЃР»РѕРІ
- truncated: РѕР±СЂС‹РІ/РЅРµР·Р°РІРµСЂС€С‘РЅРЅРѕСЃС‚СЊ
- issues: РіР»Р°РІРЅС‹Рµ РїСЂРѕР±Р»РµРјС‹, 1-2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ, РєРѕРЅРєСЂРµС‚РЅРѕ РїРѕ С„РѕРЅРµРјРµ/СЃР»РѕРІСѓ

РћС‚РІРµС‚СЊ РўРћР›Р¬РљРћ JSON-РѕР±СЉРµРєС‚РѕРј Р±РµР· markdown, СЂРѕРІРЅРѕ СЃ СЌС‚РёРјРё РєР»СЋС‡Р°РјРё:
{{"text_fidelity": int, "endings": int, "naturalness": int, "prosody": int, "accent": int, "palatalization": int, "stress": int, "artifacts": int, "truncated": bool, "words_mangled": [], "words_dropped": [], "words_misstressed": [], "issues": "string", "overall": int, "verdict": "РіРѕРґРµРЅ|РґРѕСЂР°Р±РѕС‚РєР°|Р±СЂР°Рє"}}"""


def call_model(model, b64, text, retries=3):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            {"type": "text", "text": PROMPT.format(text=text)},
        ]}],
        "max_tokens": 600,
        "temperature": 0.0,
    }
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
            content = data["choices"][0]["message"].get("content")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            txt = str(content)
            m = txt.find("{")
            n = txt.rfind("}")
            if m >= 0 and n > m:
                obj = json.loads(txt[m:n + 1])
                return obj, txt
            return None, txt[:400]
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(3)
    return None, f"ERROR {last}"


def merge_votes(objs):
    """Merge N judge verdicts: average overall, majority-filtered mangled/dropped."""
    objs = [o for o in objs if o]
    if not objs:
        return None
    import math
    def avg(key, default=0):
        vals = [o.get(key, default) for o in objs if isinstance(o.get(key), (int, float))]
        return int(round(sum(vals) / len(vals))) if vals else default
    def maj_str_list(field):
        from collections import Counter
        c = Counter()
        for o in objs:
            for s in (o.get(field) or []):
                if isinstance(s, str):
                    c[s] += 1
        return [s for s, n in c.items() if n > len(objs) / 2]
    merged = dict(objs[0])
    for k in ("text_fidelity", "endings", "naturalness", "prosody", "palatalization",
              "stress", "artifacts", "overall"):
        merged[k] = avg(k)
    for k in ("words_mangled", "words_dropped", "words_misstressed"):
        merged[k] = maj_str_list(k)
    merged["votes"] = len(objs)
    issues = [o.get("issues") for o in objs if o.get("issues")]
    merged["issues"] = issues[0] if issues else ""
    return merged


def parse_json_list_reply(txt):
    m0 = txt.find("[")
    m1 = txt.rfind("]")
    if m0 >= 0 and m1 > m0:
        try:
            return json.loads(txt[m0:m1 + 1])
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash-medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--results", default="results_gemini.json")
    args = ap.parse_args()

    man_path = os.path.join(args.dir, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8"))
    results_path = os.path.join(args.dir, args.results)
    done = {}
    if os.path.exists(results_path):
        try:
            for row in json.load(open(results_path, encoding="utf-8")):
                done[row.get("file_name")] = row
        except Exception:
            done = {}

    out = []
    n_ok = 0
    items = man[: args.limit] if args.limit else man
    for i, item in enumerate(items):
        fname = os.path.basename(item.get("file") or item.get("gen") or "")
        if fname in done:
            out.append(done[fname])
            continue
        text = item.get("text", "")
        b64 = base64.b64encode(open(item["file"] if "file" in item else item["gen"], "rb").read()).decode()
        t0 = time.time()
        objs, raws = [], []
        for _ in range(max(1, args.votes)):
            obj, raw = call_model(args.model, b64, text)
            if obj:
                objs.append(obj)
            else:
                raws.append(raw)
        obj = merge_votes(objs) if objs else None
        raw = raws[0] if raws else ""
        dt = round(time.time() - t0, 1)
        row = {"file_name": fname, "text": text, "model": args.model, "seconds": dt,
               "judge": obj if obj else None, "raw": raw[:600] if not obj else ""}
        if obj:
            n_ok += 1
        out.append(row)
        json.dump(out, open(results_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        ov = obj.get("overall") if obj else "?"
        man_ = (obj or {}).get("words_mangled") or []
        print(f"[{i + 1}/{len(items)}] {fname}: overall={ov} mangled={man_} | {dt}s", flush=True)
        if not obj:
            print("   RAW:", raw[:200], flush=True)

    ok = sum(1 for r in out if r.get("judge"))
    print(f"\nJUDGE_DONE: {ok}/{len(out)} parsed OK -> {results_path}")


if __name__ == "__main__":
    main()

