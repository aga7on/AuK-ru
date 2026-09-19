# -*- coding: utf-8 -*-
"""Добыча hard cases из артефактов прогонов s7@5750 (+ эмоции) → hard_cases_ru_v1.jsonl.

Источники:
  1. baseline_report.json (120 контроль): wer, dels/subs, flags, first_ok;
  2. judge v3 (136): words_mangled/words_dropped/issues/verdict;
  3. clone100: objective.json (wer/sim) + judge (mangled words, «sim ок — дикция плохая»);
  4. emotion judge s7@5750: брак/доработка + issues;
  5. громкость между seeds (emotion pack, seeds 7/999): RMS-разброс > 3 dB.

Выход: local_train/hard_cases/hard_cases_ru_v1.jsonl (по одной строке на уникальный текст)
       + local_train/hard_cases/HARD_CASES_V1.md (сводка).
Категории: word_drop, substitution, ending_mangle, repetition, consonant_confusion,
           sim_ok_diction_bad, loudness_seed_variance, judge_verdict_bad, phonetics_fail.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
OUT_DIR = os.path.join(AUK, "local_train", "hard_cases")
os.makedirs(OUT_DIR, exist_ok=True)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def jlines(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def clean(t):
    return (t or "").replace("+", "").strip()


def norm_key(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", (t or "").lower().replace("+", "")).strip()


cases = defaultdict(lambda: {"categories": set(), "evidence": []})


def add(text, category, evidence, source):
    k = norm_key(text)
    if not k:
        return
    c = cases[k]
    c["text"] = clean(text)
    c["categories"].add(category)
    c["evidence"].append({"source": source, **evidence})


# --- 1. baseline_report (120 контроль) ---
rep = jload(os.path.join(D, "s7_5750_report", "baseline_report.json"))
for it in rep["items"]:
    exp = it.get("expected")
    if not exp or it["kind"] == "capability":
        continue
    ev = {"wer": it.get("wer"), "dels": it.get("dels"), "subs": it.get("subs"),
          "heard": (it.get("heard") or "")[:120], "flags": it.get("flags")}
    if it.get("wer") is not None and it["wer"] > 0.15:
        add(exp, "word_drop" if it.get("dels") else "substitution", ev, f"control:{it['id']}")
    if it.get("flags"):
        add(exp, "judge_verdict_bad", ev, f"control:{it['id']}")

# --- 2. judge v3 (136): mangled/dropped words, брак ---
for r in jlines(os.path.join(D, "s7_5750_judge_results.jsonl")):
    if r.get("status") != "ok":
        continue
    j = r["judge"]
    text = clean(r.get("text") or "")
    if not text:
        continue
    ev = {"verdict": j.get("verdict"), "overall": j.get("overall"),
          "words_mangled": j.get("words_mangled"), "words_dropped": j.get("words_dropped"),
          "issues": (j.get("issues") or "")[:200]}
    if j.get("words_mangled") or j.get("words_dropped"):
        cat = "ending_mangle"
        iss = (j.get("issues") or "").lower()
        if any(w in iss for w in ("повтор", "repeat")):
            cat = "repetition"
        add(text, cat, ev, f"judge136:{r.get('task_id')}")
    if j.get("verdict") == "брак":
        add(text, "judge_verdict_bad", ev, f"judge136:{r.get('task_id')}")

# --- 3. clone100: objective + judge ---
_o = jload(os.path.join(AUK, "local_tests", "s7_5750_clone100", "objective.json"))
_o = _o["rows"] if isinstance(_o, dict) and "rows" in _o else _o
obj = {r["id"]: r for r in _o}
res100 = {r["id"]: r for r in jload(os.path.join(AUK, "local_tests", "s7_5750_clone100", "results.json"))}
for r in jlines(os.path.join(D, "s7_5750_clone100_judge_results.jsonl")):
    if r.get("status") != "ok":
        continue
    tid = r.get("task_id")
    text = clean(res100.get(tid, {}).get("text") or "")
    o = obj.get(tid, {})
    j = r["judge"]
    ev = {"wer": o.get("wer"), "sim": o.get("sim"), "verdict": j.get("verdict"),
          "words_mangled": j.get("words_mangled"), "issues": (j.get("issues") or "")[:200]}
    if o.get("wer") is not None and o["wer"] > 0.2 and (o.get("sim") or 0) >= 0.7:
        add(text, "sim_ok_diction_bad", ev, f"clone100:{tid}")
    elif o.get("wer") is not None and o["wer"] > 0.2:
        add(text, "substitution", ev, f"clone100:{tid}")
    if j.get("words_mangled"):
        add(text, "ending_mangle", ev, f"clone100:{tid}")

# --- 4. emotion judge: брак ---
emo_texts = {}
for r in jload(os.path.join(AUK, "local_tests", "emotion_ru_s7_5750", "results.json")):
    emo_texts[r["id"]] = r["text"]
for r in jlines(os.path.join(D, "emotion_s7_5750_judge_results.jsonl")):
    if r.get("status") != "ok":
        continue
    j = r["judge"]
    text = clean(emo_texts.get(r.get("task_id"), ""))
    if j.get("verdict") in ("брак",) or (j.get("words_mangled") and j.get("verdict") != "годен"):
        add(text, "judge_verdict_bad",
            {"emotion": r.get("task_id", "").split("_")[1] if r.get("task_id") else None,
             "verdict": j.get("verdict"), "issues": (j.get("issues") or "")[:200]},
            f"emotion:{r.get('task_id')}")

# --- 5. громкость между seeds (emotion pack, 7 vs 999) ---
emo_dir = os.path.join(AUK, "local_tests", "emotion_ru_s7_5750")
pairs = defaultdict(dict)
for r in jload(os.path.join(emo_dir, "results.json")):
    if r["status"] != "ok":
        continue
    base = r["id"].rsplit("_s", 1)[0]
    pairs[base][r["seed"]] = r["file"]
loud_flags = 0
for base, sd in pairs.items():
    if 7 in sd and 999 in sd:
        def rms_db(p):
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            act = x[np.abs(x) > 0.005]
            if len(act) == 0:
                return -60.0
            return float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12))
        try:
            d = abs(rms_db(sd[7]) - rms_db(sd[999]))
        except Exception:
            continue
        if d > 3.0:
            loud_flags += 1
            t = clean(emo_texts.get(base + "_s7") or emo_texts.get(base + "_s999") or "")
            add(t, "loudness_seed_variance", {"rms_diff_db": round(d, 2), "pair": base}, "emotion:seeds")

# --- write jsonl + report ---
rows = []
for i, (k, c) in enumerate(sorted(cases.items(), key=lambda kv: -len(kv[1]["categories"]))):
    rows.append({"id": f"hc_{i:03d}", "text": c["text"],
                 "categories": sorted(c["categories"]),
                 "n_evidence": len(c["evidence"]),
                 "evidence": c["evidence"][:4]})
with open(os.path.join(OUT_DIR, "hard_cases_ru_v1.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

cat_count = defaultdict(int)
for r in rows:
    for c in r["categories"]:
        cat_count[c] += 1
L = ["# HARD CASES v1 (s7@5750, добыто из реальных провалов)", "",
     f"Всего уникальных текстов: **{len(rows)}**; loudness-пар >3dB: {loud_flags}", "",
     "| категория | n |", "|---|---|"]
for c, n in sorted(cat_count.items(), key=lambda x: -x[1]):
    L.append(f"| {c} | {n} |")
L += ["", "Источник: `mine_hard_cases.py`. Файл: `hard_cases_ru_v1.jsonl`.",
      "Назначение: regression-бенчмарк S8/S9 (генерация этих текстов не должна ухудшаться)."]
open(os.path.join(OUT_DIR, "HARD_CASES_V1.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

print("cases:", len(rows))
print("categories:", dict(sorted(cat_count.items(), key=lambda x: -x[1])))
print("loudness pairs >3dB:", loud_flags)
