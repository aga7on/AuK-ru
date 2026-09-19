"""Оценка upstream-контроля (upstream base vs B@500) через Gemini-судью v3.

Строит manifest для audit_gemini_judge_v3.py (--manifest mode): для каждой из 31 задач
оригинального AuK — упрощённая проверка «выполнена ли задача» (task prompt).
usage: python upstream_judge_manifest.py --out manifest.json
"""
import json
import os

AUK = r"G:\AI\AuK"
UP = os.path.join(AUK, "local_tests", "upstream_control")

CHECKS = {
    "zero-shot tts": "речь на английском тем же голосом, что референс; текст совпадает",
    "instruct tts": "речь на китайском, соответствует описанию голоса/стиля",
    "content editing": "указанная правка текста выполнена, остальное сохранено",
    "acoustic editing": "указанное акустическое изменение выполнено",
    "paralinguistic": "указанное паралингвистическое изменение (эмоция/шёпот/дыхание) выполнено",
    "enhancement & separation": "улучшение/разделение выполнено, речь сохранена",
}


def checks_for(task, instr):
    t = task.lower()
    for k, v in CHECKS.items():
        if k in t:
            return v
    return "целевой эффект достигнут; содержание сохранено"


def main():
    out = []
    for variant in ("upstream", "B500"):
        res = json.load(open(os.path.join(UP, variant, "results.json"), encoding="utf-8"))
        for r in res["results"]:
            if r["status"] != "ok":
                continue
            out.append({
                "id": f"{variant}__{r['id']}", "mode": "tool",
                "text": r["task"], "instruction": r["instruction"],
                "goal": f"выполнить задачу '{r['task']}' оригинального AuK",
                "checks": checks_for(r["task"], r["instruction"]),
                "ref": r["ref"], "file": r["file"],
            })
    p = os.path.join(AUK, "local_train", "reports", "deepseek_supervised",
                     "upstream_judge_manifest.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {p}")


if __name__ == "__main__":
    main()
