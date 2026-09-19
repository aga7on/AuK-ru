"""Манифест судьи v3 для s7-эмоций (group=clone, схема CLONE — валидна).
Судья проверяет голос/текст/чистоту; эмоцию измеряет Aniemore-гейт отдельно.
Использование:
  python emotion_ru_judge_manifest_s7.py --src <dir с results.json> --out <manifest.json>
"""
import argparse
import json
import os

AUK = r"G:\AI\AuK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with results.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = json.load(open(os.path.join(args.src, "results.json"), encoding="utf-8"))
    out = []
    for r in results:
        if r["status"] != "ok":
            continue
        out.append({
            "id": r["id"], "mode": "clone", "group": "clone",
            "instruction": f"Reproduce the reference voice and say in Russian with a {r['emotion']} tone: '{r['text']}'",
            "text": r["text"],
            "goal": f"Русская речь голосом референса с эмоцией {r['emotion']}",
            "checks": (f"целевая эмоция '{r['emotion']}' явно выражена в интонации/энергии; "
                       "голос совпадает с референсом; произношение чёткое, русский акцент нейтральный; "
                       "артефакты, шёпот/крик вместо целевой эмоции — брак"),
            "ref": r["ref"], "file": r["file"], "wav": r["file"],
            "emotion": r["emotion"],
        })
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {args.out}")


if __name__ == "__main__":
    main()