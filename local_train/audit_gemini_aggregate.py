"""Aggregate blind Gemini v2 results by variant (joins SECRET_map; automated signal only)."""
import csv
import json
import os
from collections import defaultdict

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "exploratory")


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def rate(flags):
    return round(sum(1 for f in flags if f) / len(flags), 3) if flags else None


def main():
    listen = {r["file"]: r for r in csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8"))}
    secret = {r["file"]: r["variant"] for r in csv.DictReader(open(os.path.join(BLIND, "SECRET_map.csv"), encoding="utf-8"))}
    rows = {}
    for line in open(os.path.join(BLIND, "results_audit_gemini_v2.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["file"]] = r

    variants = ["u10000", "A@250", "A@500", "B@250", "B@500"]
    per = {v: defaultdict(list) for v in variants}
    status_count = defaultdict(lambda: defaultdict(int))
    for f, r in rows.items():
        v = secret.get(f)
        if not v:
            continue
        j = r.get("judge") or {}
        g = r.get("group")
        per[v]["n"].append(1)
        status_count[v][r["status"]] += 1
        if r["status"] != "ok":
            continue
        per[v]["overall"].append(j.get("overall"))
        if g in ("tts", "phonetics", "phrase_training"):
            per[v]["fid"].append(j.get("text_fidelity"))
            per[v]["nat"].append(j.get("naturalness"))
            per[v]["sim"].append(j.get("voice_similarity_to_ref"))
            per[v]["subs"].append(len(j.get("phoneme_substitutions") or []))
            per[v]["trunc"].append(bool(j.get("truncated")))
        elif g == "clone":
            per[v]["fid"].append(j.get("text_fidelity"))
            per[v]["sim"].append(j.get("voice_similarity_to_ref"))
            per[v]["leak"].append(bool(j.get("ref_word_leak")))
        else:
            per[v]["op"].append(j.get("operation_performed"))
            per[v]["match"].append(bool(j.get("expected_effect_matches")))
            per[v]["cont"].append(j.get("content_preserved"))
            per[v]["art"].append(bool(j.get("introduced_artifacts")))

    L = []
    A = L.append
    A("# blind_s2 — автосудья Gemini v2 по вариантам (EXPLORATORY, не решение)")
    A("")
    A("Источник: `local_tests/blind_s2/results_audit_gemini_v2.jsonl` (685/685 ok),")
    A("prompt `audit-2026-09-16-v2`, model `gemini-3.8-flash-medium`. Варианты вскрыты через")
    A("`SECRET_map.csv` ТОЛЬКО после завершения автоматической оценки. Это вспомогательный")
    A("сигнал: слепое прослушивание не проведено, решение по вариантам не принято.")
    A("")
    A("| вариант | n | overall | fid | nat | sim | sub/файл | trunc | leak | op | match | cont | art |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        d = per[v]
        A(f"| {v} | {len(d['n'])} | {mean(d['overall'])} | {mean(d['fid'])} | {mean(d['nat'])} | "
          f"{mean(d['sim'])} | {mean(d['subs'])} | {rate(d['trunc'])} | {rate(d['leak'])} | "
          f"{mean(d['op'])} | {rate(d['match'])} | {mean(d['cont'])} | {rate(d['art'])} |")
    A("")
    A("Колонки: fid — точность текста; nat — естественность; sim — сходство с референсом;")
    A("sub/файл — среднее число фонетических замен; trunc/leak — доля; op — выполнение операции;")
    A("match — доля соответствия ожидаемому эффекту; cont — сохранность содержания; art — доля")
    A("внесённых дефектов. Для clone колонки nat/sub/op/match/cont/art не применимы (пусто).")
    A("")
    A("## Статусы")
    A("")
    A("| вариант | ok | no_result | invalid_schema | invalid_json | error |")
    A("|---|---:|---:|---:|---:|---:|")
    for v in variants:
        s = status_count[v]
        A(f"| {v} | {s.get('ok',0)} | {s.get('no_result',0)} | {s.get('invalid_schema',0)} | "
          f"{s.get('invalid_json',0)} | {s.get('error',0)} |")
    A("")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "blind_s2_gemini_v2_by_variant.md")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("written", p)
    print("\n".join(L))


if __name__ == "__main__":
    main()
