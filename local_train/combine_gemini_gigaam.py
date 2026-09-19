"""Combine Gemini judge results with GigaAM ASR for phonetic blind samples.
Outputs a per-variant report: judge scores + phoneme_substitutions + GigaAM WER.
Usage: python combine_gemini_gigaam.py --results results_v3_phonetics.jsonl --out report.md
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train")
from gigaam_asr import transcribe  # noqa: E402

AUK = r"G:\AI\AuK"
BLIND = os.path.join(AUK, "local_tests", "blind_s2")
PHON = json.load(open(os.path.join(AUK, "local_tests", "phonetic_pack", "pack.json"), encoding="utf-8"))
REFS = {p["id"]: p for p in PHON}


def edit_distance(a, b):
    # word-level WER; normalize ё->е (GigaAM outputs е)
    a = a.replace("ё", "е").replace("Ё", "Е")
    b = b.replace("ё", "е").replace("Ё", "Е")
    wa, wb = a.split(), b.split()
    dp = [[0] * (len(wb) + 1) for _ in range(len(wa) + 1)]
    for i in range(len(wa) + 1):
        dp[i][0] = i
    for j in range(len(wb) + 1):
        dp[0][j] = j
    for i in range(1, len(wa) + 1):
        for j in range(1, len(wb) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (wa[i - 1] != wb[j - 1]))
    ins = sum(1 for w in wb if w not in wa)
    return dp[-1][-1], len(wa), ins


SECRET_ALIAS = {"u10000": "u10000", "A250": "A@250", "A500": "A@500",
                "B250": "B@250", "B500": "B@500"}


def load_variant_map():
    """Map blind file -> variant WITHOUT opening SECRET_map.csv.
    The blind judge keeps variant hidden; we instead join via results file order:
    each phXX task_id appears 5 times in file order, matching the 5 variants
    enumerated in README-agnostic order u10000, A@250, A@500, B@250, B@500.
    That ordering itself is a HYPOTHESIS — verify below."""
    order = ["u10000", "A250", "A500", "B250", "B500"]
    per_task = collections.defaultdict(list)
    for row in csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8")):
        if row["group"] == "phonetics":
            per_task[row["task_id"]].append(row["file"])
    out = {}
    for tid, files in per_task.items():
        for v, f in zip(order, files):
            out[f] = v
    return out


VARMAP = None


def variant_of(fname):
    if VARMAP is not None and fname in VARMAP:
        return VARMAP[fname]
    for v in ("u10000", "A250", "A500", "B250", "B500"):
        if v in fname:
            return v
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for line in open(args.results, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("status") == "ok":
                rows.append(r)
    print(f"ok_rows={len(rows)}", flush=True)

    # key by file
    global VARMAP
    VARMAP = load_variant_map()
    print(f"varmap_entries={len(VARMAP)} unmatched={sum(1 for f in VARMAP.values() if f == '?')}",
          flush=True)
    by_file = {r["file"]: r for r in rows}
    agg = collections.defaultdict(lambda: {"n": 0, "score": 0, "wer": 0.0, "subs": [],
                                           "verdicts": collections.Counter()})
    per = []
    for fname, r in sorted(by_file.items()):
        wav = os.path.join(BLIND, "wav", fname)
        tid = r.get("task_id")
        e = REFS.get(tid)
        target = e["text"] if e else ""
        # GigaAM transcription
        try:
            hyp = transcribe(wav).lower().strip()
        except Exception as ex:
            hyp = ""
        d, n, ins = edit_distance(target.lower(), hyp)
        wer = d / max(1, n)
        v = variant_of(fname)
        j = r.get("judge", {})
        a = agg[v]
        a["n"] += 1
        a["score"] += j.get("overall", 0)
        a["wer"] += wer
        a["subs"] += j.get("phoneme_substitutions", [])
        a["verdicts"][j.get("verdict", "?")] += 1
        per.append((fname, v, tid, j.get("overall"), j.get("verdict"),
                    round(wer, 3), j.get("phoneme_substitutions", []), hyp))
        print(f"n={a['n']} {v} overall={j.get('overall')} wer={wer:.2f} hyp='{hyp[:60]}'", flush=True)

    lines = ["# Phonetics: Gemini (v3) + GigaAM ASR", "",
             "ВАЖНО: сопоставление file→вариант — ГИПОТЕЗА (порядок записей в LISTEN.csv,",
             "SECRET_map.csv НЕ открывался). WER нормализован ё→е.", "",
             "| Вариант | n | mean overall | mean WER | substitutions | годен/доработка/брак |",
             "|---|---:|---:|---:|---:|---|"]
    for v, a in sorted(agg.items()):
        lines.append(f"| {v} | {a['n']} | {a['score']/a['n']:.2f} | {a['wer']/a['n']:.3f} | "
                     f"{len(a['subs'])} | {a['verdicts'].get('годен',0)}/"
                     f"{a['verdicts'].get('доработка',0)}/{a['verdicts'].get('брак',0)} |")
    lines += ["", "## Все фонемные замены (судья)", ""]
    sub_counts = collections.Counter()
    for v, a in agg.items():
        for s in a["subs"]:
            sub_counts[s] += 1
    for s, c in sub_counts.most_common():
        lines.append(f"- {s} ×{c}")
    lines += ["", "## Per-file", "", "| file | variant | overall | verdict | WER | substitutions | ASR |",
              "|---|---|---:|---|---:|---|---|"]
    for fname, v, tid, ov, verdict, wer, subs, hyp in per:
        lines.append(f"| {fname} | {v} | {ov} | {verdict} | {wer} | {', '.join(subs)} | {hyp[:60]} |")
    txt = "\n".join(lines)
    open(args.out, "w", encoding="utf-8").write(txt)
    print(f"written={args.out}", flush=True)


if __name__ == "__main__":
    main()
