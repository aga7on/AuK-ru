# -*- coding: utf-8 -*-
"""S10: automatic candidate mining → preference dataset {prompt, ref, chosen, rejected}.

Источники (существующие артефакты, GPU не нужен):
  1. seed-пробы best-of-3 (tmp_seed_probe_s7_5750, …_s8_7750): 25 клонов × 3 seeds,
     метрика WeSpeaker sim → chosen=argmax, rejected=argmin (если gap ≥ 0.03);
  2. эмо-паки (emotion_ru_s7_5750/6750, emotion_ru_s8_7750): 30 (voice,emo,phrase) × 2 seeds,
     метрика judge overall/verdict → chosen/rejected (если verdict различается или Δoverall ≥ 2);
  3. clone100 (s7_5750_clone100, s8_7750_clone100): один кандидат — нет пар (пропускаем,
     но фиксируем как будущий источник для N>1 майнинга).

Правила качества пар (anti-noise):
  - сим-пары: gap ≥ 0.03, обе стороны не error;
  - judge-пары: обе записи status=ok; приоритет verdict (годен > доработка > брак),
    внутри одного verdict — Δoverall ≥ 2.

Выход: local_train/preference/pairs_v1.jsonl + PREFERENCE_V1.md (свод по источникам).
Формат строки:
 {"pair_id", "source", "prompt_id", "ref", "chosen", "rejected",
  "chosen_score", "rejected_score", "metric", "text"}
"""
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
OUT_DIR = os.path.join(AUK, "local_train", "preference")
os.makedirs(OUT_DIR, exist_ok=True)

VERD_RANK = {"годен": 2, "доработка": 1, "брак": 0}


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


pairs = []


def add(source, pid, ref, chosen, rejected, cs, rs, metric, text=""):
    pairs.append({"pair_id": f"{source}:{pid}", "source": source, "prompt_id": pid,
                  "ref": ref, "chosen": chosen, "rejected": rejected,
                  "chosen_score": cs, "rejected_score": rs, "metric": metric, "text": text})


# --- 1) seed probes ---
for tag in ("tmp_seed_probe_s7_5750", "tmp_seed_probe_s8_7750"):
    sp = jd(os.path.join(AUK, "local_tests", tag, "results.json")) or []
    for r in sp:
        sims = r.get("sims") or []
        if len(sims) < 2:
            continue
        seeds = (7, 123, 999)
        best_i = max(range(len(sims)), key=lambda i: sims[i])
        worst_i = min(range(len(sims)), key=lambda i: sims[i])
        if sims[best_i] - sims[worst_i] < 0.03 or best_i == worst_i:
            continue
        base = os.path.join(AUK, "local_tests", tag)
        add(tag, r["id"], r.get("ref"),
            os.path.join(base, f"{r['id']}_s{seeds[best_i]}.wav"),
            os.path.join(base, f"{r['id']}_s{seeds[worst_i]}.wav"),
            sims[best_i], sims[worst_i], "wespeaker_sim")

# --- 2) emotion packs (judge) ---
EMO_PACKS = {
    "emotion_ru_s7_5750": "emotion_s7_5750_judge_results.jsonl",
    "emotion_ru_s7_6750": "emotion_s7_6750_judge_results.jsonl",
    "emotion_ru_s8_7750": "emotion_s8_7750_judge_results.jsonl",
}
for pack, jf in EMO_PACKS.items():
    res = jd(os.path.join(AUK, "local_tests", pack, "results.json")) or []
    by_base = defaultdict(dict)
    for r in res:
        if r.get("status") != "ok":
            continue
        base = r["id"].rsplit("_s", 1)[0]
        by_base[base][r["seed"]] = r
    judged = {r.get("task_id"): r["judge"] for r in jl(os.path.join(AUK, "local_train", "reports", "deepseek_supervised", jf)) if r.get("status") == "ok"}
    for base, sd in by_base.items():
        if len(sd) < 2:
            continue
        seeds = sorted(sd)
        js = [(s, judged.get(f"{base}_s{s}")) for s in seeds]
        js = [(s, j) for s, j in js if j]
        if len(js) < 2:
            continue
        def keyf(sj):
            return (VERD_RANK.get(sj[1].get("verdict"), -1), sj[1].get("overall", 0))
        best = max(js, key=keyf)
        worst = min(js, key=keyf)
        if keyf(best) == keyf(worst):
            continue
        if best[1].get("verdict") == worst[1].get("verdict") and abs(best[1].get("overall", 0) - worst[1].get("overall", 0)) < 2:
            continue
        rb, rw = sd[best[0]], sd[worst[0]]
        add(pack, base, rb.get("ref"), rb.get("file"), rw.get("file"),
            best[1].get("overall"), worst[1].get("overall"), "judge_overall", rb.get("text", ""))

# --- write ---
with open(os.path.join(OUT_DIR, "pairs_v1.jsonl"), "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

by_src = Counter(p["source"] for p in pairs)
# integrity: все файлы существуют
missing = sum(1 for p in pairs if not (os.path.exists(p["chosen"]) and os.path.exists(p["rejected"])))

L = ["# PREFERENCE v1 — automatic candidate mining (S10 prep)", "",
     f"Пар: **{len(pairs)}** (missing files: {missing} — должно быть 0)", "",
     "| источник | пар | метрика |", "|---|---|---|"]
for s, n in by_src.most_common():
    metric = next(p["metric"] for p in pairs if p["source"] == s)
    L.append(f"| {s} | {n} | {metric} |")
L += ["", "Правила: sim-пары gap ≥ 0.03; judge-пары — различие verdict или Δoverall ≥ 2.",
      "Назначение: seed для S10 preference/rejection-обучения (first-shot reliability).",
      "Расширение: N>1 майнинг на clone100-промптах (4–8 кандидатов) — следующий шаг S10."]
open(os.path.join(OUT_DIR, "PREFERENCE_V1.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

print("pairs:", len(pairs), dict(by_src), "missing:", missing)
