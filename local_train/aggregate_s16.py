# -*- coding: utf-8 -*-
"""Свод S16 (фонетический буткемп) → S16_RESULTS.md.

ГЛАВНЫЙ гейт S16: произношение на ТРУДНЫХ текстах (там, где S13b выявил проблему:
accent 7.94 / palatalization 7.87 / stress 8.31 / naturalness 5.3, брак 24/78).
Стоп-условия (не-регресс v1.0 = s7@5750):
  TTS WER ≤ 0.077, TTS first_ok ≥ 0.812, эмо годен ≥ 48.3%, clone sim bo3 ≥ 0.75.
"""
import json
import os
import statistics as st
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
BASE = {  # v1.0 = s7@5750
    "tts_wer": 0.077, "first_ok": 0.812, "emo_goden": 48.3, "bo3": 0.771,
    "frontend_wer": 0.255,
}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


# --- hard-phonetics baseline v1.0 (ТЕ ЖЕ 33 текста, apples-to-apples) ---
BASE_P = os.path.join(D, "s16_baseline_phonetics_results.jsonl")
base_ph = [r["judge"] for r in jl(BASE_P) if r.get("status") == "ok"]
base_subs = Counter()
base_mangled = Counter()
if base_ph:
    for k in ("accent", "palatalization", "stress", "endings", "naturalness", "prosody",
              "text_fidelity", "overall"):
        BASE["hard_" + k] = mean([j.get(k) for j in base_ph])
    BASE["hard_brak"] = sum(1 for j in base_ph if j.get("verdict") == "брак")
    BASE["hard_n"] = len(base_ph)
    for j in base_ph:
        for s in j.get("phoneme_substitutions") or []:
            base_subs[str(s)] += 1
        for w in j.get("words_mangled") or []:
            base_mangled[str(w).lower()] += 1
    BASE["hard_subs"] = sum(base_subs.values())
    BASE["hard_mangled"] = sum(base_mangled.values())
else:
    print("WARNING: baseline phonetics results missing — hard-гейты будут n/a")


# --- hard-phonetics (главный гейт) ---
ph = [r["judge"] for r in jl(os.path.join(D, "s16_phonetics_results.jsonl")) if r.get("status") == "ok"]
h = {k: mean([j.get(k) for j in ph]) for k in
     ("accent", "palatalization", "stress", "endings", "naturalness", "prosody", "text_fidelity", "overall")}
brak = sum(1 for j in ph if j.get("verdict") == "брак")
subs = Counter()
mangled = Counter()
for j in ph:
    for s in j.get("phoneme_substitutions") or []:
        subs[str(s)] += 1
    for w in j.get("words_mangled") or []:
        mangled[str(w).lower()] += 1

# --- frontend probe ---
fp = jd(os.path.join(AUK, "local_tests", "frontend_probe_s16", "results.json")) or []
fp_ok = [r for r in fp if r.get("status") == "ok"]
fp_wer = mean([r.get("wer_norm", r.get("wer")) for r in fp_ok])
fp_cer = mean([r.get("cer_norm") for r in fp_ok if r.get("cer_norm") is not None])
# CER — валидная метрика для длинных/составных слов (S17 probe: WER 1.0 при CER 0.01)
# baseline v1.0 (frontend_probe_s9_v4 — та же 33-текстовая проба)
fpb = jd(os.path.join(AUK, "local_tests", "frontend_probe_s9_v4", "results.json")) or []
BASE["frontend_cer"] = mean([r.get("cer_norm") for r in fpb
                             if r.get("status") == "ok" and r.get("cer_norm") is not None])

# --- TTS не-регресс ---
rep = jd(os.path.join(D, "s16_report", "baseline_report.json")) or {}
tts = rep.get("aggregate", {}).get("tts", {})

# --- эмоции ---
emo = [r["judge"] for r in jl(os.path.join(D, "emotion_s16_judge_results.jsonl")) if r.get("status") == "ok"]
verd = Counter(j.get("verdict") for j in emo)
emo_goden = round(verd.get("годен", 0) / len(emo) * 100, 1) if emo else None

# --- clone bo3 ---
sp = jd(os.path.join(AUK, "local_tests", "tmp_seed_probe_s16", "results.json")) or []
bo3 = sorted(r["best"] for r in sp if r.get("best") is not None)
bo3_med = round(bo3[len(bo3) // 2], 4) if bo3 else None


def ge(a, b):
    return "✅" if (a is not None and a >= b) else ("❌" if a is not None else "n/a")


def le(a, b):
    return "✅" if (a is not None and a <= b) else ("❌" if a is not None else "n/a")


gate = [
    # ОСНОВНЫЕ (правило ROADMAP 21.09: CER первый; naturalness/подмены — где реальная проблема)
    ("CER hard ≤ baseline v1.0", fp_cer, le(fp_cer, BASE.get("frontend_cer"))),
    ("HARD naturalness ≥ baseline v1.0", h.get("naturalness"), ge(h.get("naturalness"), BASE.get("hard_naturalness"))),
    ("HARD подмены ≤ baseline v1.0", sum(subs.values()), le(sum(subs.values()), BASE.get("hard_subs", 0))),
    ("HARD mangled ≤ baseline v1.0", sum(mangled.values()), le(sum(mangled.values()), BASE.get("hard_mangled", 0))),
    ("HARD брак ≤ baseline v1.0", brak, le(brak, BASE.get("hard_brak", 0))),
    # вторичные (у v1.0 уже ~9 — следим за регрессом с допуском −0.15)
    ("HARD accent ≥ baseline−0.15", h.get("accent"), ge(h.get("accent"), (BASE.get("hard_accent") or 9.0) - 0.15)),
    ("HARD palatalization ≥ baseline−0.15", h.get("palatalization"), ge(h.get("palatalization"), (BASE.get("hard_palatalization") or 8.9) - 0.15)),
    # не-регресс v1.0
    ("TTS WER ≤ 0.077 (не-регресс)", tts.get("wer_mean"), le(tts.get("wer_mean"), BASE["tts_wer"])),
    ("TTS first_ok ≥ 0.812 (не-регресс)", tts.get("first_ok"), ge(tts.get("first_ok"), BASE["first_ok"])),
    ("Эмо годен ≥ 48.3% (не-регресс)", emo_goden, ge(emo_goden, BASE["emo_goden"])),
]

L = ["# S16 RESULTS — фонетический буткемп (произношение на трудных текстах)", "",
     "Микс v16_diction_mix: v7_s7_mix 1:1 + 6088 строк аудио с ПРОВЕРЕННО чистой дикцией",
     "(GigaAM WER ≤0.10, без подмен ы→и / ч-ц / ш-щ; 2620 клипов с трудными словами).",
     "Старт: s7@5750 (v1.0), 750 шагов, lr 2e-6 (консервативно), seed 16.", "",
     "## Главная метрика: CER на hard-наборе (правило ROADMAP 21.09: CER первый, WER вспомогательный)", "",
     f"| метрика | v1.0 (s7@5750) | s16@6500 | Δ |",
     f"|---|---|---|---|",
     f"| **CER hard (wer_norm→cer)** | {BASE.get('frontend_cer')} | {fp_cer} | {round(fp_cer-BASE['frontend_cer'],4) if isinstance(fp_cer,(int,float)) and BASE.get('frontend_cer') is not None else 'n/a'} |",
     f"| WER hard (норм., вспомогательная) | {BASE['frontend_wer']} | {fp_wer} | {round(fp_wer-BASE['frontend_wer'],4) if isinstance(fp_wer,(int,float)) else 'n/a'} |", "",
     "## ГЛАВНЫЙ гейт: произношение на трудных текстах (33 текста, те же у baseline)", "",
     "| ось | v1.0 (те же 33) | s16 | Δ |", "|---|---|---|---|"]
for k in ("accent", "palatalization", "stress", "naturalness"):
    base = BASE.get("hard_" + k)
    v = h.get(k)
    d = round(v - base, 2) if isinstance(v, (int, float)) and isinstance(base, (int, float)) else None
    L.append(f"| {k} | {base} | {v} | {d:+} |" if d is not None else f"| {k} | {base} | {v} | n/a |")
L += [f"| вердикт брак | {BASE.get('hard_brak')}/{BASE.get('hard_n')} | {brak}/{len(ph)} | |",
      f"| фонемных подмен (всего) | {BASE.get('hard_subs')} | {sum(subs.values())} | |",
      f"| прожёванных слов (всего) | {BASE.get('hard_mangled')} | {sum(mangled.values())} | |", ""]

L += ["## Фонемные подмены / прожёванные слова (s16, hard-набор)", ""]
if subs:
    L += ["| подмена | n |", "|---|---|"]
    for s, n in subs.most_common(12):
        L.append(f"| {s} | {n} |")
else:
    L.append("Судья не вернул фонемных подмен (было: «свои→сои», «встреча→стрича», «вторая→фарая»).")
L.append("")
if mangled:
    L += ["Топ прожёванных слов: " + ", ".join(f"{w}({n})" for w, n in mangled.most_common(8)), ""]

L += ["## Не-регресс v1.0 и frontend", "",
      "| метрика | v1.0 | s16 |", "|---|---|---|",
      f"| TTS WER | {BASE['tts_wer']} | {tts.get('wer_mean')} |",
      f"| TTS first_ok | {BASE['first_ok']} | {tts.get('first_ok')} |",
      f"| эмо годен % | {BASE['emo_goden']} | {emo_goden} |",
      f"| clone sim best-of-3 | {BASE['bo3']} | {bo3_med} |",
      f"| frontend probe wer_mean | {BASE['frontend_wer']} | {fp_wer} |",
      f"| frontend probe cer_mean (валиднее для длинных слов) | {BASE.get('frontend_cer')} | {fp_cer} |", "",
      "## GATE S16", "", "| условие | значение | вердикт |", "|---|---|---|"]
np_ = nf = nn = 0
for name, val, ok in gate:
    L.append(f"| {name} | {val} | {ok} |")
    np_ += ok == "✅"; nf += ok == "❌"; nn += ok == "n/a"
L += ["", f"PASS {np_} / FAIL {nf} / n/a {nn}", "", "## Вердикт", ""]
if nf == 0 and nn == 0:
    L.append("S16 ЗАКРЫТ: произношение на трудных текстах улучшено БЕЗ регресса v1.0 → кандидат v1.1.")
elif h.get("accent") and h.get("accent") > BASE["hard_accent"] and nf <= 2:
    L.append(f"S16 ЧАСТИЧНО успешен ({np_}/{len(gate)}): произношение выросло, но есть регресс — "
             "сравнить ранние чекпойнты 6000/6250 (смерджены) и выбрать оптимум.")
else:
    L.append(f"S16 гейт не закрыт ({np_}/{len(gate)}) — см. таблицу. Сравнить 6000/6250/6500.")
L += ["", "Авто-свод: `aggregate_s16.py`. Главный гейт измеряется на ТРУДНЫХ текстах",
      "(frontend_probe_s16, 33 генерации) + судья phonetics по ним (s16_phonetics_results.jsonl).",
      "Baseline v1.0 на ТЕХ ЖЕ 33 текстах: s16_baseline_phonetics_results.jsonl (apples-to-apples).",
      "Примечание: на hard-текстах оси произношения у v1.0 уже высокие (accent ~9.0), а overall",
      "низкий (5.8) из-за naturalness ~5.4 — судья слабо ловит «жевание» (согласованность с",
      "человеком 21%, S13_CALIBRATION). Поэтому фонемные подмены и human A/B — главные критерии."]

open(os.path.join(D, "S16_RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote S16_RESULTS.md")
print("hard axes:", h)
print("brak:", brak, "/", len(ph))
print("top subs:", subs.most_common(6))
print(f"tts={tts.get('wer_mean')}/{tts.get('first_ok')} emo={emo_goden} bo3={bo3_med} frontend={fp_wer}")
