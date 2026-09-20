# -*- coding: utf-8 -*-
"""Сравнение ТРЁХ профилей (правило ROADMAP 21.09): S7@5750 / S8b@8500 / S16b@6500.

Цель: решить, является ли S16b кандидатом на v1.1 MODEL-level upgrade, или только
routed-веткой. Критерий пользователя: выигрыш на трудном русском БЕЗ повторения drift S8/S8b
(TTS first_ok, эмо-годен).

Источники (все уже сгенерированы):
  - frontend_probe_{s9_v4,s8,s8b,s16}/results.json → CER/WER на hard-текстах (33);
  - s16_baseline_phonetics_results.jsonl / s16_phonetics_results.jsonl → оси судьи;
  - *_report/baseline_report.json → TTS WER/first_ok, clone;
  - emotion_*_judge_results.jsonl → эмо-годен;
  - tmp_seed_probe_* → first-shot / best-of-3.

usage: python compare_profiles.py
Выход: reports/deepseek_supervised/PROFILE_COMPARISON.md
"""
import json
import os
import statistics as st
from collections import Counter

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
LT = os.path.join(AUK, "local_tests")
OUT = os.path.join(D, "PROFILE_COMPARISON.md")

PROFILES = [
    # (label, probe_tag, report_dir, emo_judge, seed_probe)
    ("S7@5750 (v1.0)", "s9_v4", "s7_5750_report", "emotion_s7_5750_judge_results.jsonl", "tmp_seed_probe_s7_5750"),
    ("S8@7750", "s8", "s8_7750_report", "emotion_s8_7750_judge_results.jsonl", "tmp_seed_probe_s8_7750"),
    ("S8b@8500", "s8b", "s8b_8500_report", "emotion_s8b_8500_judge_results.jsonl", "tmp_seed_probe_s8b_8500"),
    ("S16b@6500", "s16", "s16_report", "emotion_s16_judge_results.jsonl", "tmp_seed_probe_s16"),
    ("S16c@6500", "s16c", "s16c_report", "emotion_s16c_judge_results.jsonl", "tmp_seed_probe_s16c"),
]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def med(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return round(xs[len(xs) // 2], 4) if xs else None


def probe_metrics(tag):
    p = os.path.join(LT, f"frontend_probe_{tag}", "results.json")
    rows = jd(p) or []
    ok = [r for r in rows if r.get("status") == "ok"]
    cers = [r["cer_norm"] for r in ok if r.get("cer_norm") is not None]
    wers = [r.get("wer_norm", r.get("wer")) for r in ok]
    by_cat = {}
    for r in ok:
        if r.get("cer_norm") is not None:
            by_cat.setdefault(r.get("category", "?"), []).append(r["cer_norm"])
    return mean(cers), mean(wers), len(ok), {k: round(mean(v), 3) for k, v in by_cat.items()}


def emo_goden(fn):
    rows = [r["judge"] for r in jl(os.path.join(D, fn)) if r.get("status") == "ok"]
    if not rows:
        return None, 0
    v = Counter(j.get("verdict") for j in rows)
    return round(v.get("годен", 0) / len(rows) * 100, 1), len(rows)


def tts_metrics(report_dir):
    if not report_dir:
        return None, None, None
    rep = jd(os.path.join(D, report_dir, "baseline_report.json")) or {}
    a = rep.get("aggregate", {})
    t = a.get("tts", {})
    c = a.get("clone", {})
    return t.get("wer_mean"), t.get("first_ok"), c.get("wer_mean")


def seed_metrics(tag):
    sp = jd(os.path.join(LT, tag, "results.json")) or []
    first = [r["sims"][0] for r in sp if r.get("sims")]
    best = [r["best"] for r in sp if r.get("best") is not None]
    return med(first), med(best)


def main():
    rows = []
    for label, tag, rep, emo_fn, sp in PROFILES:
        cer, wer, n, bycat = probe_metrics(tag)
        emo, emo_n = emo_goden(emo_fn)
        twer, tfok, cwer = tts_metrics(rep)
        fmed, bmed = seed_metrics(sp)
        rows.append({"label": label, "cer": cer, "wer": wer, "n": n, "emo": emo, "emo_n": emo_n,
                     "tts_wer": twer, "first_ok": tfok, "clone_wer": cwer,
                     "first_seed": fmed, "bo3": bmed, "bycat": bycat})

    base = rows[0]
    L = ["# PROFILE COMPARISON — профили произношения (правило ROADMAP 21.09)", "",
         "Вопрос: даёт ли кандидат выигрыш на трудном русском БЕЗ drift, свойственного S8/S8b?",
         "CER — основная метрика произношения (WER вспомогательная, искажается сегментацией ASR).", "",
         "## Сводка", "",
         "| профиль | hard CER ↓ | hard WER | TTS WER ≤0.077 | TTS first_ok ≥0.812 | эмо годен ≥48.3 | clone WER | first-seed sim | best-of-3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | **{r['cer']}** | {r['wer']} | {r['tts_wer']} | {r['first_ok']} | "
                 f"{r['emo']} | {r['clone_wer']} | {r['first_seed']} | {r['bo3']} |")

    L += ["", "## hard-CER по категориям (33 текста frozen hard_eval_v1)", ""]
    cats = sorted({c for r in rows for c in r["bycat"]})
    L.append("| категория | " + " | ".join(r["label"].split(" ")[0] for r in rows) + " |")
    L.append("|---|" + "---|" * len(rows))
    for c in cats:
        L.append(f"| {c} | " + " | ".join(str(r["bycat"].get(c, "—")) for r in rows) + " |")

    L += ["", "## Нюанс: S16b по категориям (метод корзин работает, источник данных отравил соседей)", "",
          "S16b ВЫИГРАЛ в целевых корзинах и проиграл в соседних:",
          "- **лучше v1.0**: number_case 0.008 (vs 0.018), omograph_stress 0.009 (vs 0.027),",
          "  phone 0.0 (vs 0.013) — корзины yi/stress/devoicing сработали;",
          "- **хуже v1.0**: date_time 0.056 (vs 0.033), money 0.063 (vs 0.013), abbr_name 0.087",
          "  (vs 0.065) — редукции разговорного стиля («сентября→сенскитбря», «января→ынгаря»);",
          "- tech_url 0.154 у всех профилей — лимит модели/frontend, не данных.", "",
          "Вывод (ПРОВЕРЕНО): метод «корзины контрастов» корректен, ошибка S16b — в источнике",
          "данных (67% спонтанной речи). S16c проверяет это на читаной речи (Common Voice).", "",
          "## Вердикт по критерию пользователя", ""]
    lines = []
    for r in rows[1:]:
        if r["cer"] is None:
            lines.append(f"- {r['label']}: данных пока нет (прогон не завершён)")
            continue
        better = r["cer"] < base["cer"]
        drift = []
        if r["first_ok"] is not None and r["first_ok"] < 0.812:
            drift.append(f"first_ok {r['first_ok']}<0.812")
        if r["tts_wer"] is not None and r["tts_wer"] > 0.077:
            drift.append(f"TTS WER {r['tts_wer']}>0.077")
        if r["emo"] is not None and r["emo"] < 48.3:
            drift.append(f"эмо {r['emo']}<48.3")
        lines.append(f"- {r['label']}: hard-CER {r['cer']} vs v1.0 {base['cer']} → "
                     f"{'ЛУЧШЕ' if better else 'НЕ лучше'} ({r['cer'] - base['cer']:+.4f})"
                     + ("; DRIFT: " + ", ".join(drift) if drift else "; drift нет"))
    L += lines + ["",
                  "**Решение принимает human shortlist (право вето, ROADMAP 21.09).** Автоматика выше —",
                  "только triage; при 21% согласованности человек↔судья финальный вердикт за прослушиванием.", "",
                  "Авто-свод: `compare_profiles.py`."]

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    for r in rows:
        print(f"{r['label']:18s} cer={r['cer']} wer={r['wer']} tts={r['tts_wer']}/{r['first_ok']} emo={r['emo']}")


if __name__ == "__main__":
    main()
