# -*- coding: utf-8 -*-
"""Свод S12 (Emotion v2): эмо-судья + intensity + whisper + TTS-регресс → S12_RESULTS.md.

Gate S12 (S12_DESIGN.md):
  эмо-годен ≥ 55% (v1.0: 48.3%), sad ≥ 7/12 (v1.0: 4/12),
  intensity monotonic any_rate ≥ 0.70, whisper ΔRMS ≤ −2 дБ (v1.0: +0.2/+0.1 — FAIL),
  TTS WER ≤ 0.077 и first_ok ≥ 0.812 (не хуже v1.0).
"""
import json
import os
import statistics as st
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# --- эмо-судья ---
emo = [r["judge"] for r in jl(os.path.join(D, "emotion_s12_7250_judge_results.jsonl")) if r.get("status") == "ok"]
emo_ids = [r.get("task_id", "") for r in jl(os.path.join(D, "emotion_s12_7250_judge_results.jsonl")) if r.get("status") == "ok"]
verd = Counter(j.get("verdict") for j in emo)
emo_goden = round(verd.get("годен", 0) / len(emo) * 100, 1) if emo else None
by_emo = defaultdict(lambda: [0, 0])
for tid, j in zip(emo_ids, emo):
    e = next((x for x in ("happy", "sad", "angry", "fearful", "excited") if f"_{x}_" in tid), "?")
    by_emo[e][1] += 1
    if j.get("verdict") == "годен":
        by_emo[e][0] += 1

# --- intensity ---
ip = jd(os.path.join(AUK, "local_tests", "intensity_s12_7250", "results.json")) or []
mono_any = sum(1 for r in ip if r.get("mono_any")) / len(ip) if ip else None
mono_rms = sum(1 for r in ip if r.get("mono_rms")) / len(ip) if ip else None
d_rms_med = st.median([r["d_rms"] for r in ip]) if ip else None

# --- whisper ---
wp = jd(os.path.join(AUK, "local_tests", "whisper_probe_s12", "results.json")) or []
byw = defaultdict(list)
for r in wp:
    byw[r["variant"]].append(r["rms_db"])
whisper_delta = None
if byw.get("normal") and (byw.get("whisper_mode") or byw.get("whisper_tone")):
    wm = byw.get("whisper_mode") or byw.get("whisper_tone")
    whisper_delta = round(st.median(wm) - st.median(byw["normal"]), 1)

# --- TTS регресс ---
rep = jd(os.path.join(D, "s12_7250_report", "baseline_report.json")) or {}
tts = rep.get("aggregate", {}).get("tts", {})

rows_gate = [
    ("Эмо-годен ≥ 55% (v1.0: 48.3)", emo_goden, "ge", 55.0),
    ("sad годен ≥ 7/12 (v1.0: 4)", by_emo.get("sad", [0, 0])[0], "ge", 7),
    ("intensity monotonic ≥ 0.70", mono_any, "ge", 0.70),
    ("whisper ΔRMS ≤ −2 дБ (v1.0: +0.2)", whisper_delta, "le", -2.0),
    ("TTS WER ≤ 0.077", tts.get("wer_mean"), "le", 0.077),
    ("TTS first_ok ≥ 0.812", tts.get("first_ok"), "ge", 0.812),
]


def verdict(val, op, thr):
    if val is None:
        return "n/a"
    ok = (val >= thr) if op == "ge" else (val <= thr)
    return "✅" if ok else "❌"


L = ["# S12 RESULTS — Emotion v2 (intensity + speaking_mode)", "",
     "Микс: v12_emo_mix (v7 + intensity-тертили low/mid/high + speaking_mode whisper/laughing).",
     "Старт: s7@5750 (v1.0), 1500 шагов lr 3e-6, seed 12 → model_7250.", "",
     "## Показатели", "",
     f"- эмо-годен (судья, 60): **{emo_goden}%** {dict(verd)} (v1.0: 48.3%)",
     f"- годен по эмоциям: " + ", ".join(f"{e} {g}/{n}" for e, (g, n) in sorted(by_emo.items())),
     f"- intensity monotonic: any {mono_any}, rms {mono_rms}, ΔRMS median {d_rms_med} dB (n={len(ip)})",
     f"- whisper ΔRMS vs normal: {whisper_delta} dB (v1.0: +0.2 tone / +0.1 mode — не шёпот)",
     f"- TTS WER {tts.get('wer_mean')} / first_ok {tts.get('first_ok')} (v1.0: 0.077/0.812)", "",
     "## GATE S12", "", "| условие | значение | вердикт |", "|---|---|---|"]
n_pass = n_fail = 0
for name, val, op, thr in rows_gate:
    v = verdict(val, op, thr)
    n_pass += v == "✅"
    n_fail += v == "❌"
    L.append(f"| {name} | {val} | {v} |")
L += ["", f"PASS {n_pass} / FAIL {n_fail} / n/a {len(rows_gate)-n_pass-n_fail}", "",
      "## Вердикт", ""]
if n_fail == 0:
    L.append("S12 ЗАКРЫТ: эмоции v2 работают → кандидат v1.1 (полный GATES-прогон обязателен перед каноном).")
else:
    L.append("S12 гейт не закрыт полностью — см. строки ❌; ранние чекпойнты 6500/7000 смерджены, "
             "возможен выбор оптимума (урок s7).")
L += ["", "Авто-свод: `aggregate_s12.py`. Артефакты: `emotion_ru_s12_7250`, `intensity_s12_7250`, "
      "`whisper_probe_s12` (v1.0 baseline: `whisper_probe_v1`), `s12_7250_control/report`."]

open(os.path.join(D, "S12_RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote S12_RESULTS.md")
print(f"emo_goden={emo_goden} mono_any={mono_any} whisper_delta={whisper_delta} tts={tts.get('wer_mean')}/{tts.get('first_ok')}")
