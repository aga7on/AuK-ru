"""Манифест судьи v3 для emotion_ru (инструктивные эмоции, s5@4500).
Судья оценивает: выражена ли ЦЕЛЕВАЯ эмоция, сохранился ли голос, произношение, артефакты.
"""
import json
import os

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_tests", "emotion_ru")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "emotion_ru_judge_manifest.json")


def main():
    results = json.load(open(os.path.join(SRC, "results.json"), encoding="utf-8"))
    out = []
    for r in results:
        if r["status"] != "ok":
            continue
        out.append({
            "id": r["id"], "mode": "clone", "group": "capability",
            "instruction": r["instruction"] if "instruction" in r else
            f"Reproduce the reference voice and say in Russian with a {r['emotion']} tone: '{r['text']}'",
            "text": r["text"],
            "goal": f"Русская речь голосом референса с эмоцией {r['emotion']}",
            "checks": (f"целевая эмоция '{r['emotion']}' явно выражена в интонации/энергии; "
                       "голос совпадает с референсом; произношение чёткое, русский акцент нейтральный; "
                       "артефакты, шёпот/крик вместо целевой эмоции — брак"),
            "ref": r["ref"], "file": r["file"], "wav": r["file"],
        })
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
