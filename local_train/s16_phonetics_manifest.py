# -*- coding: utf-8 -*-
"""S16: манифест судьи v3 (group=phonetics, mode=tts, ref=None) для hard-генераций.

Источник: <probe dir>/results.json (33 генерации на трудных текстах hard_cases_ru —
тот же набор текстов, что и у baseline v1.0, поэтому сравнение apples-to-apples).
Главный гейт S16: судья оценивает именно произношение (оси accent/palatalization/
stress/endings/phoneme_substitutions), а не «китайский акцент» словами (гейт невалиден).

usage:
  python s16_phonetics_manifest.py --src local_tests\frontend_probe_s16 --out <manifest>
  python s16_phonetics_manifest.py --src local_tests\frontend_probe_s9_v4 --out <baseline manifest>
"""
import argparse
import json
import os

AUK = r"G:\AI\AuK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(AUK, "local_tests", "frontend_probe_s16"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="s16", help="метка варианта в id (s16 / s7_baseline)")
    args = ap.parse_args()

    rows = json.load(open(os.path.join(args.src, "results.json"), encoding="utf-8"))
    out = []
    miss = 0
    for r in rows:
        if r.get("status") != "ok" or not r.get("file") or not os.path.exists(r["file"]):
            miss += 1
            continue
        out.append({
            "id": f"{r['id']}__{r['category']}__{args.tag}",
            "mode": "tts",            # mode = ПРОМПТ (TTS_NOREF при ref=None)
            "group": "phonetics",     # group = СХЕМА валидации (ERRORS.MD #6)
            "text": r["expected"],
            "ref": None,
            "file": r["file"],
            "wav": r["file"],
            "category": r["category"],
            "goal": "Чистое русское произношение трудного текста",
            "checks": ("нейтральный русский акцент; мягкие согласные перед и/е/ё/ю/я произнесены "
                       "мягко; «ы» не заменяется на «и»; ударения на месте; окончания не проглочены; "
                       "числа/даты произнесены словами полностью"),
        })
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} (missing: {miss}) -> {args.out}")


if __name__ == "__main__":
    main()
