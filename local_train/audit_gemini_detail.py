"""Grouped per-task-kind and per-operation analysis of the blind Gemini results.

- uses v2 records for all groups, replaced by v3 records for clone (semantic leak fix);
- per variant (5), per group and per operation: valid/total denominators and key metrics;
- matched paired differences by task_id (not a cross-task average used for selection);
- phoneme-substitution / accent / stress / naturalness distributions and worst files.
Writes GEMINI_DETAIL.md. Automated signal only; human listening is optional corroboration.
"""
import csv
import json
import os
from collections import defaultdict

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
REP = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
V2 = os.path.join(BLIND, "results_audit_gemini_v2.jsonl")
V3C = os.path.join(REP, "results_audit_gemini_v3_clone.jsonl")
OUT = os.path.join(REP, "GEMINI_DETAIL.md")
VARIANTS = ["u10000", "A@250", "A@500", "B@250", "B@500"]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def rate(flags):
    flags = list(flags)
    return round(sum(1 for f in flags if f) / len(flags), 3) if flags else None


def dedupe(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                d[r["file"]] = r
    return d


def main():
    listen = {r["file"]: r for r in csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8"))}
    secret = {r["file"]: r["variant"] for r in csv.DictReader(open(os.path.join(BLIND, "SECRET_map.csv"), encoding="utf-8"))}
    v2, v3c = dedupe(V2), dedupe(V3C)

    rows = []
    for f, meta in listen.items():
        r = v3c.get(f) if meta["group"] == "clone" and v3c.get(f, {}).get("status") == "ok" else v2.get(f)
        if not r or r.get("status") != "ok":
            continue
        rows.append({"file": f, "variant": secret[f], "group": meta["group"],
                     "task_id": meta["task_id"], "j": r["judge"],
                     "src": "v3" if (meta["group"] == "clone" and r is v3c.get(f)) else "v2"})
    print("rows", len(rows), "clone_src_v3", sum(1 for r in rows if r["src"] == "v3"))

    def metric(r, key):
        return r["j"].get(key)

    def leak(r):
        return r["j"].get("ref_content_leak", r["j"].get("ref_word_leak"))

    GRP_KEYS = {
        "tts": ["overall", "text_fidelity", "naturalness", "voice_similarity_to_ref", "accent", "stress"],
        "phonetics": ["overall", "text_fidelity", "naturalness", "voice_similarity_to_ref", "accent", "stress"],
        "phrase_training": ["overall", "text_fidelity", "naturalness"],
        "clone": ["overall", "text_fidelity", "voice_similarity_to_ref", "naturalness"],
        "tool": ["overall", "operation_performed", "content_preserved", "quality"],
        "capability": ["overall", "operation_performed", "content_preserved", "quality"],
    }
    L = ["# Слепой набор: детально по вариантам и операциям (EXPLORATORY, авто-сигнал)", "",
         "Источник: v2 (все группы) + v3 (clone, исправленная семантика утечки). Варианты вскрыты",
         "через SECRET_map ПОСЛЕ слепой оценки. Это не решение: слепое прослушивание опционально.",
         "Кросс-задачных средних для ВЫБОРА не используется; ниже — по-задачные/по-операционным и парные разницы.", ""]

    def variant_table(title, filt, keys):
        L.append(f"## {title}")
        L.append("")
        L.append("| вариант | valid/total | " + " | ".join(keys) + " |")
        L.append("|---|" + "---:|" * (len(keys) + 1))
        tot = defaultdict(int)
        val = defaultdict(int)
        agg = {v: defaultdict(list) for v in VARIANTS}
        for r in rows:
            if not filt(r):
                continue
            tot[r["variant"]] += 1
            for k in keys:
                x = metric(r, k)
                if isinstance(x, (int, float)):
                    agg[r["variant"]][k].append(x)
            if all(isinstance(metric(r, k), (int, float)) for k in keys):
                val[r["variant"]] += 1
        for v in VARIANTS:
            L.append(f"| {v} | {val[v]}/{tot[v]} | " + " | ".join(str(mean(agg[v][k])) for k in keys) + " |")
        L.append("")

    for g in ["tts", "phonetics", "phrase_training", "clone", "tool", "capability"]:
        variant_table(f"Группа: {g}", lambda r, g=g: r["group"] == g, GRP_KEYS[g])

    # per-operation for tool and capability
    for g in ["tool", "capability"]:
        ops = sorted({r["task_id"].rsplit("_", 1)[0] if g == "tool" else r["task_id"]
                      for r in rows if r["group"] == g})
        L.append(f"## По-операционные таблицы: {g}")
        for op in ops:
            sub = [r for r in rows if r["group"] == g and (r["task_id"].startswith(op + "_") or r["task_id"] == op)]
            if not sub:
                continue
            L.append(f"### {op}")
            L.append("")
            L.append("| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for v in VARIANTS:
                rs = [r for r in sub if r["variant"] == v]
                L.append(f"| {v} | {sum(1 for r in rs if isinstance(metric(r,'overall'),(int,float)))}/{len(rs)} | "
                         f"{mean([metric(r,'overall') for r in rs])} | {mean([metric(r,'operation_performed') for r in rs])} | "
                         f"{rate([metric(r,'expected_effect_matches') for r in rs])} | "
                         f"{mean([metric(r,'content_preserved') for r in rs])} | "
                         f"{rate([metric(r,'introduced_artifacts') for r in rs])} |")
            L.append("")

    # phoneme substitutions distribution
    L.append("## Фонетические замены (tts+phonetics+phrase)")
    L.append("")
    L.append("| вариант | файлов с sub>0 | всего sub | sub/файл |")
    L.append("|---|---:|---:|---:|")
    for v in VARIANTS:
        vals = [len(r["j"].get("phoneme_substitutions") or []) for r in rows
                if r["variant"] == v and r["group"] in ("tts", "phonetics", "phrase_training")]
        L.append(f"| {v} | {sum(1 for x in vals if x > 0)} | {sum(vals)} | {mean(vals)} |")
    L.append("")

    # leak rates (clone)
    L.append("## Утечка содержимого референса (clone)")
    L.append("")
    L.append("| вариант | leak rate |")
    L.append("|---|---:|")
    for v in VARIANTS:
        L.append(f"| {v} | {rate([leak(r) for r in rows if r['group']=='clone' and r['variant']==v])} |")
    L.append("")

    # paired differences by task_id
    L.append("## Парные разницы по task_id (значение варианта минус u10000)")
    L.append("")
    L.append("| группа | метрика | сравнение | mean delta | лучше/хуже/равно |")
    L.append("|---|---|---|---:|---|")
    by_task = defaultdict(dict)
    for r in rows:
        by_task[(r["group"], r["task_id"])][r["variant"]] = r
    for g in ["tts", "phonetics", "clone", "tool", "capability"]:
        key = "overall"
        for v in ["A@500", "B@500"]:
            ds = []
            for (gg, tid), d in by_task.items():
                if gg != g or "u10000" not in d or v not in d:
                    continue
                a, b = metric(d["u10000"], key), metric(d[v], key)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    ds.append(b - a)
            if ds:
                win = sum(1 for x in ds if x > 0)
                lose = sum(1 for x in ds if x < 0)
                eq = len(ds) - win - lose
                L.append(f"| {g} | {key} | {v} - u10000 | {mean(ds)} | {win}/{lose}/{eq} |")
    L.append("")

    # representative failing files
    L.append("## Худшие файлы (overall, по группам и вариантам; top-5)")
    for g in ["tts", "phonetics", "clone", "tool", "capability"]:
        for v in ["u10000", "A@500", "B@500"]:
            sub = sorted([r for r in rows if r["group"] == g and r["variant"] == v],
                         key=lambda r: metric(r, "overall") if isinstance(metric(r, "overall"), (int, float)) else 99)
            worst = sub[:5]
            L.append(f"- **{g} / {v}**: " + ", ".join(f"{r['task_id']}({r['file']}={metric(r,'overall')})" for r in worst))
    L.append("")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
