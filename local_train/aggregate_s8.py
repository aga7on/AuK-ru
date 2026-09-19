# -*- coding: utf-8 -*-
"""Свод S8: агрегирует objective + judge артефакты → S8_RESULTS.md (сравнение s5/s7/s8).

Gate S8 (ROADMAP): эмоции ≥ S7 (годен 48.3%), TTS ≥ S7 (WER ≤0.077, first_ok ≥0.812),
neutral clone/phonetics судья ≥ S5 (clone 8.20, phonetics 8.06).
Читает только уже готовые файлы; недостающие помечает n/a (не падает).
"""
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


# --- baseline objective (s8) ---
rep = jd(os.path.join(D, "s8_7750_report", "baseline_report.json"))
agg = (rep or {}).get("aggregate", {})
tts = agg.get("tts", {})
clone = agg.get("clone", {})

# --- clone100 objective ---
c100 = jd(os.path.join(AUK, "local_tests", "s8_7750_clone100", "objective.json"))
c100_rows = c100 if isinstance(c100, list) else (c100 or {}).get("rows", [])
sims = sorted(r["sim"] for r in c100_rows if isinstance(r.get("sim"), (int, float)))
wers = sorted(r["wer"] for r in c100_rows if isinstance(r.get("wer"), (int, float)))
c100_sim_med = round(sims[len(sims) // 2], 4) if sims else None
c100_wer_med = round(wers[len(wers) // 2], 4) if wers else None

# --- seed probe best-of-3 ---
sp = jd(os.path.join(AUK, "local_tests", "tmp_seed_probe_s8_7750", "results.json")) or []
bo3 = sorted(r["best"] for r in sp if r.get("best") is not None)
bo3_med = round(bo3[len(bo3) // 2], 4) if bo3 else None
first = sorted(r["sims"][0] for r in sp if r.get("sims"))
first_med = round(first[len(first) // 2], 4) if first else None

# --- judges ---
def judge_group(path):
    rows = jl(path)
    g = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            g[r.get("group", "?")].append(r["judge"].get("overall"))
    return {k: mean(v) for k, v in g.items()}, len(rows)


main_g, main_n = judge_group(os.path.join(D, "s8_7750_judge_results.jsonl"))
emo_rows = jl(os.path.join(D, "emotion_s8_7750_judge_results.jsonl"))
emo_ok = [r["judge"] for r in emo_rows if r.get("status") == "ok"]
emo_verd = Counter(r.get("verdict") for r in emo_ok)
emo_goden = round(emo_verd.get("годен", 0) / len(emo_ok) * 100, 1) if emo_ok else None

# --- frontend probe ---
fp = jd(os.path.join(AUK, "local_tests", "frontend_probe_s8", "results.json")) or []
fp_ok = [r for r in fp if r.get("status") == "ok"]
fp_wer = mean([r.get("wer_norm", r.get("wer")) for r in fp_ok])

# --- DSP (read TOOL_DSP.md row for s8@7750 if present) ---
dsp = "см. TOOL_DSP.md"

# --- reference values (s5, s7 from prior runs) ---
REF = {
    "s5@4500": {"tts_wer": 0.081, "first_ok": 0.750, "clone_sim_bo3": 0.763, "clone_wer": 0.055,
                "judge_clone": 8.20, "judge_phon": 8.06, "emo_goden": 36.7},
    "s7@5750": {"tts_wer": 0.077, "first_ok": 0.812, "clone_sim_bo3": 0.771, "clone_wer": 0.079,
                "judge_clone": 7.56, "judge_phon": 7.44, "emo_goden": 48.3},
}

s8 = {"tts_wer": tts.get("wer_mean"), "first_ok": tts.get("first_ok"),
      "clone_sim_bo3": bo3_med, "clone_sim_1seed": clone.get("sim_median"),
      "clone_wer": clone.get("wer_mean"), "c100_sim_med": c100_sim_med, "c100_wer_med": c100_wer_med,
      "first_seed_med": first_med, "judge_tts": main_g.get("tts"), "judge_clone": main_g.get("clone"),
      "judge_phon": main_g.get("phonetics"), "judge_tool": main_g.get("tool"),
      "emo_goden": emo_goden, "frontend_wer": fp_wer}


def ge(a, b):
    return "✅" if (a is not None and b is not None and a >= b) else ("❌" if a is not None and b is not None else "n/a")


def le(a, b):
    return "✅" if (a is not None and b is not None and a <= b) else ("❌" if a is not None and b is not None else "n/a")


gate = []
gate.append(("Эмоции годен ≥ S7 (48.3%)", s8["emo_goden"], ge(s8["emo_goden"], 48.3)))
gate.append(("TTS WER ≤ S7 (0.077)", s8["tts_wer"], le(s8["tts_wer"], 0.077)))
gate.append(("TTS first_ok ≥ S7 (0.812)", s8["first_ok"], ge(s8["first_ok"], 0.812)))
gate.append(("judge clone ≥ S5 (8.20)", s8["judge_clone"], ge(s8["judge_clone"], 8.20)))
gate.append(("judge phonetics ≥ S5 (8.06)", s8["judge_phon"], ge(s8["judge_phon"], 8.06)))

L = ["# S8 RESULTS — neutral recovery (ROADMAP S8)", "",
     "Сравнение s5@4500 (fallback neutral) / s7@5750 (v1.0 канон) / s8@7750 (этот прогон).", "",
     "## Метрики", "",
     "| метрика | s5@4500 | s7@5750 | s8@7750 |", "|---|---|---|---|",
     f"| TTS WER (mean) | 0.081 | 0.077 | {s8['tts_wer']} |",
     f"| TTS first_ok | 0.750 | 0.812 | {s8['first_ok']} |",
     f"| clone sim 1-seed (median) | 0.742 | 0.741 | {s8['clone_sim_1seed']} |",
     f"| clone sim best-of-3 (median) | 0.763 | 0.771 | {s8['clone_sim_bo3']} |",
     f"| first-seed median (seed-probe) | — | 0.7378 | {s8['first_seed_med']} |",
     f"| clone WER (mean) | 0.055 | 0.079 | {s8['clone_wer']} |",
     f"| clone100 sim median | 0.711 | 0.703 | {s8['c100_sim_med']} |",
     f"| clone100 WER median | 0.143 | 0.143 | {s8['c100_wer_med']} |",
     f"| judge tts | 6.90 | 7.06 | {s8['judge_tts']} |",
     f"| judge clone | 8.20 | 7.56 | {s8['judge_clone']} |",
     f"| judge phonetics | 8.06 | 7.44 | {s8['judge_phon']} |",
     f"| judge tool | 8.11 | 7.74 | {s8['judge_tool']} |",
     f"| эмо годен % | 36.7 | 48.3 | {s8['emo_goden']} |",
     f"| frontend probe wer_mean | — | 0.255 | {s8['frontend_wer']} |",
     f"| DSP | 6/6 | 6/6 | {dsp} |", "",
     "## GATE S8", "", "| условие | значение | вердикт |", "|---|---|---|"]
for name, val, ok in gate:
    L.append(f"| {name} | {val} | {ok} |")
passed = sum(1 for _, _, ok in gate if ok == "✅")
failed = sum(1 for _, _, ok in gate if ok == "❌")
na = sum(1 for _, _, ok in gate if ok == "n/a")
L += ["", f"PASS {passed} / FAIL {failed} / n/a {na}", "",
      "## Вердикт", "",
      ("S8 закрыл гейт — кандидат на новый канон (neutral recovery без потери эмоций)."
       if failed == 0 and na == 0 else
       "S8 НЕ закрыл все гейты — см. таблицу; решение: дообучение / откат к s7@5750 / точечная правка микса."),
      "", "Авто-свод: `aggregate_s8.py`. Артефакты: `s8_7750_*`, `emotion_ru_s8_7750`, `tmp_seed_probe_s8_7750`, `frontend_probe_s8`."]

out = os.path.join(D, "S8_RESULTS.md")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", out)
print(json.dumps(s8, ensure_ascii=False))
print("gate:", [(g[0], g[2]) for g in gate])
