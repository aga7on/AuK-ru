"""Judge manifest для 100-клонового стресс-теста s5 (судья v3).
Проверки: нейтральный русский акцент (НЕ 'китайский'), сходство голоса, артефакты, разборчивость.
"""
import json
import os

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "s5_clone100_judge_manifest.json")


def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AUK, "local_tests", "s5_clone100")
    out_name = sys.argv[2] if len(sys.argv) > 2 else "s5_clone100_judge_manifest.json"
    OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", out_name)
    results = json.load(open(os.path.join(src, "results.json"), encoding="utf-8"))
    out = []
    for r in results:
        if r["status"] != "ok":
            continue
        out.append({
            "id": r["id"], "mode": "clone", "group": "clone",
            "instruction": r["instruction"], "text": r["text"],
            "goal": "Клон голоса референса на русском языке",
            "checks": "произношение чёткое, нейтральный русский акцент (не иностранный/'китайский'), "
                      "сходство тембра с референсом, отсутствие артефактов, полное воспроизведение текста",
            "ref": r["ref"], "file": r["file"], "wav": r["file"],
        })
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
