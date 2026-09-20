# -*- coding: utf-8 -*-
"""Свод S15: композиция адаптеров (emotion s7@5750 + neutral s5@4500 α=0.5) → S15_RESULTS.md.

Gate S15 (S15_DESIGN.md): эмоции ≥48.3% (v1.0), judge clone ≥8.20 (s5), judge phonetics ≥8.06 (s5),
TTS WER ≤0.077, first_ok ≥0.812 (не хуже v1.0).
"""
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


rep = jd(os.path.join(D, "s15_comp_report", "baseline_report.json")) or {}
agg = rep.get("aggregate", {})
tts = agg.get("tts", {})
clone = agg.get("clone", {})

sp = jd(os.path.join(AUK, "local_tests", "tmp_seed_probe_s15_comp", "results.json")) or []
bo3 = sorted(r["best"] for r in sp if r.get("best") is not None)
bo3_med = round(bo3[len(bo3) // 2], 4) if bo3 else None

g = defaultdict(list)
for r in jl(os.path.join(D, "s15_comp_judge_results.jsonl")):
    if r.get("status") == "ok":
        g[r["group"]].append(r["judge"].get("overall"))
judge = {k: mean(v) for k, v in g.items()}

emo = [r["judge"] for r in jl(os.path.join(D, "emotion_s15_comp_judge_results.jsonl")) if r.get("status") == "ok"]
verd = Counter(j.get("verdict") for j in emo)
emo_goden = round(verd.get("годен", 0) / len(emo) * 100, 1) if emo else None

c = {
    "tts_wer": tts.get("wer_mean"), "first_ok": tts.get("first_ok"),
    "clone_sim_1seed": clone.get("sim_median"), "clone_sim_bo3": bo3_med,
    "clone_wer": clone.get("wer_mean"),
    "judge_tts": judge.get("tts"), "judge_clone": judge.get("clone"),
    "judge_phon": judge.get("phonetics"), "judge_tool": judge.get("tool"),
    "emo_goden": emo_goden,
}

gate = []


def ge(a, b): return "✅" if (a is not None and a >= b) else ("❌" if a is not None else "n/a")
def le(a, b): return "✅" if (a is not None and a <= b) else ("❌" if a is not None else "n/a")


gate.append(("Эмоции годен ≥ 48.3 (v1.0)", c["emo_goden"], ge(c["emo_goden"], 48.3)))
gate.append(("judge clone ≥ 8.20 (s5)", c["judge_clone"], ge(c["judge_clone"], 8.20)))
gate.append(("judge phonetics ≥ 8.06 (s5)", c["judge_phon"], ge(c["judge_phon"], 8.06)))
gate.append(("TTS WER ≤ 0.077 (v1.0)", c["tts_wer"], le(c["tts_wer"], 0.077)))
gate.append(("TTS first_ok ≥ 0.812 (v1.0)", c["first_ok"], ge(c["first_ok"], 0.812)))

REF = {
    "s5@4500": dict(tts_wer=0.081, first_ok=0.750, bo3=0.763, clone_wer=0.055, judge_clone=8.20, judge_phon=8.06, emo=36.7),
    "s7@5750 (v1.0)": dict(tts_wer=0.077, first_ok=0.812, bo3=0.771, clone_wer=0.079, judge_clone=7.56, judge_phon=7.44, emo=48.3),
}

L = ["# S15 RESULTS — композиция адаптеров (emotion α1.0 + neutral α0.5)", "",
     "База: auk_ru_10000. Файл: run_s15/composed_e1.0_n0.5.safetensors.",
     "Parity-инвариант ПРОВЕРЕН: compose(α=1) ≡ merge_lora (maxdiff 1.16e-10).", "",
     "| метрика | s5@4500 | s7@5750 (v1.0) | s15 comp |", "|---|---|---|---|",
     f"| TTS WER | 0.081 | 0.077 | {c['tts_wer']} |",
     f"| TTS first_ok | 0.750 | 0.812 | {c['first_ok']} |",
     f"| clone sim best-of-3 | 0.763 | 0.771 | {c['clone_sim_bo3']} |",
     f"| clone WER | 0.055 | 0.079 | {c['clone_wer']} |",
     f"| judge clone | 8.20 | 7.56 | {c['judge_clone']} |",
     f"| judge phonetics | 8.06 | 7.44 | {c['judge_phon']} |",
     f"| judge tts | 6.90 | 7.06 | {c['judge_tts']} |",
     f"| judge tool | 8.11 | 7.74 | {c['judge_tool']} |",
     f"| эмо годен % | 36.7 | 48.3 | {c['emo_goden']} |", "",
     "## GATE S15", "", "| условие | значение | вердикт |", "|---|---|---|"]
for name, val, ok in gate:
    L.append(f"| {name} | {val} | {ok} |")
p = sum(1 for _, _, ok in gate if ok == "✅")
f = sum(1 for _, _, ok in gate if ok == "❌")
na = sum(1 for _, _, ok in gate if ok == "n/a")
L += ["", f"PASS {p} / FAIL {f} / n/a {na}", "", "## Вердикт", ""]
if f == 0 and na == 0:
    L.append("Композиция ЗАКРЫЛА trade-off: эмоции v1.0 + neutral s5 в одних весах — кандидат v1.1.")
elif p >= 3:
    L.append(f"Композиция частично успешна ({p}/5). Следующий шаг: сетка α/β {{0.25,0.5,0.75}}² — найти точку с полным закрытием.")
else:
    L.append("Композиция не даёт синергии (интерференция адаптеров). Альтернатива: маршрутизация адаптеров по инструкции (S15, вариант 3).")
L += ["", "Авто-свод: `aggregate_s15.py`; композиция: `compose_adapters.py` (parity PASS)."]

open(os.path.join(D, "S15_RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote S15_RESULTS.md")
print("metrics:", json.dumps(c, ensure_ascii=False))
print("gate:", [(g0[0][:30], g0[2]) for g0 in gate])
