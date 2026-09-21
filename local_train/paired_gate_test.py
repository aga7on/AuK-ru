# -*- coding: utf-8 -*-
"""Статистическая валидность гейтов: парные тесты v1.0 vs кандидат на ОДНИХ И ТЕХ ЖЕ элементах.

Зачем (21.09): гейты S8/S8b/S10/S16b/S16c сравнивали агрегаты на n=33/48/60. При n=60 и p≈0.48
95% CI для доли ≈ ±12 п.п. — т.е. «эмо 48.3 → 43.3» и «эмо 51.7» статистически неразличимы.
Поскольку оба прогона измеряются на ТОМ ЖЕ наборе элементов, корректен ПАРНЫЙ тест
(McNemar для бинарных, парный bootstrap/t для непрерывных) — он значительно чувствительнее.

Считает:
  - hard-CER (frontend_probe, 33 текста) — парно по текстам;
  - оси произношения судьи (s16*_phonetics_results, 33) — парно по task_id;
  - TTS WER / first_ok (baseline_report items, kind=tts, 48) — парно по id;
  - эмо-годен (emotion_*_judge_results, 60) — McNemar по id.

usage: python paired_gate_test.py [--base_tag s9_v4 --cand_tag s16c ...]
Выход: stdout + reports/deepseek_supervised/GATE_STATISTICS.md
"""
import argparse
import json
import os
import random
from collections import Counter

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
LT = os.path.join(AUK, "local_tests")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired_stats(pairs, n_boot=2000, seed=7):
    """pairs: list of (base, cand). Возвращает mean diff, 95% CI (bootstrap), долю улучшений."""
    d = [c - b for b, c in pairs if isinstance(b, (int, float)) and isinstance(c, (int, float))]
    if len(d) < 3:
        return None
    mean = sum(d) / len(d)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(len(d))] for _ in d]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    better = sum(1 for x in d if x > 1e-9)
    worse = sum(1 for x in d if x < -1e-9)
    return {"n": len(d), "mean_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": not (lo <= 0 <= hi),
            "better": better, "worse": worse, "same": len(d) - better - worse}


def mcnemar(pairs):
    """pairs: list of (base_bool, cand_bool). Возвращает b, c и приближённый p (binomial)."""
    b = sum(1 for x, y in pairs if (not x) and y)      # base fail -> cand ok
    c = sum(1 for x, y in pairs if x and (not y))      # base ok -> cand fail
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": None, "significant": False}
    # двусторонний биномиальный тест против p=0.5 (нормальное приближение + точность для малых n)
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    p = min(1.0, p)
    return {"b": b, "c": c, "n_disc": n, "p": round(p, 4), "significant": p < 0.05}


def probe_map(tag):
    rows = jd(os.path.join(LT, f"frontend_probe_{tag}", "results.json")) or []
    return {r["id"]: r for r in rows if r.get("status") == "ok"}


def judge_map(fn):
    out = {}
    for r in jl(os.path.join(D, fn)):
        if r.get("status") == "ok":
            key = r.get("task_id", "")
            out[key.split("__")[0]] = r["judge"]
    return out


def report_items(report_dir):
    rep = jd(os.path.join(D, report_dir, "baseline_report.json")) or {}
    return {r["id"]: r for r in rep.get("items", [])}


def emo_map(fn):
    out = {}
    for r in jl(os.path.join(D, fn)):
        if r.get("status") == "ok":
            out[r.get("task_id", "")] = r["judge"].get("verdict")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_probe", default="s9_v4")
    ap.add_argument("--base_judge", default="s16_baseline_phonetics_results.jsonl")
    ap.add_argument("--base_report", default="s7_5750_report")
    ap.add_argument("--base_emo", default="emotion_s7_5750_judge_results.jsonl")
    ap.add_argument("--cands", default="s16,s16c,s8b")
    args = ap.parse_args()

    CAND = {
        "s16": dict(probe="s16", judge="s16_phonetics_results.jsonl", report="s16_report",
                    emo="emotion_s16_judge_results.jsonl", label="S16b@6500"),
        "s16c": dict(probe="s16c", judge="s16c_phonetics_results.jsonl", report="s16c_report",
                     emo="emotion_s16c_judge_results.jsonl", label="S16c@6500"),
        "s8b": dict(probe="s8b", judge=None, report="s8b_8500_report",
                    emo="emotion_s8b_8500_judge_results.jsonl", label="S8b@8500"),
    }

    bp = probe_map(args.base_probe)
    bj = judge_map(args.base_judge)
    br = report_items(args.base_report)
    be = emo_map(args.base_emo)

    L = ["# GATE STATISTICS — парные тесты и шум измерения (21.09)", "",
         "Проблема: гейты сравнивали агрегаты на малых n. При n=60 и p≈0.48 95% CI доли ≈ ±12 п.п.,",
         "при n=48 для first_ok ≈ ±11 п.п. — различия в 3–5 п.п. НЕ отличимы от шума.",
         "Поскольку прогоны измеряются на ОДНИХ И ТЕХ ЖЕ элементах, применён парный тест",
         "(bootstrap 95% CI для непрерывных, McNemar exact для бинарных).", "",
         "Обозначения: Δ = кандидат − v1.0; «значимо» = CI не содержит 0 (или McNemar p<0.05).", ""]

    for tag in [c for c in args.cands.split(",") if c.strip()]:
        cfg = CAND.get(tag)
        if not cfg:
            continue
        L += [f"## {cfg['label']} vs v1.0 (s7@5750)", ""]

        # 1) hard CER / WER — парно по текстам
        cp = probe_map(cfg["probe"])
        ids = sorted(set(bp) & set(cp))
        if ids:
            cer = paired_stats([(bp[i].get("cer_norm"), cp[i].get("cer_norm")) for i in ids])
            wer = paired_stats([(bp[i].get("wer_norm", bp[i].get("wer")),
                                 cp[i].get("wer_norm", cp[i].get("wer"))) for i in ids])
            if cer:
                L.append(f"- **hard CER** (n={cer['n']}): Δ {cer['mean_diff']:+.4f}, "
                         f"CI95 [{cer['ci95'][0]:+.4f}, {cer['ci95'][1]:+.4f}] → "
                         f"{'ЗНАЧИМО' if cer['significant'] else 'в пределах шума'} "
                         f"(лучше/хуже/равно {cer['better']}/{cer['worse']}/{cer['same']})")
            if wer:
                L.append(f"- hard WER (n={wer['n']}): Δ {wer['mean_diff']:+.4f}, "
                         f"CI95 [{wer['ci95'][0]:+.4f}, {wer['ci95'][1]:+.4f}] → "
                         f"{'ЗНАЧИМО' if wer['significant'] else 'в пределах шума'}")

        # 2) оси произношения судьи — парно
        if cfg["judge"]:
            cj = judge_map(cfg["judge"])
            jids = sorted(set(bj) & set(cj))
            if jids:
                for axis in ("accent", "palatalization", "naturalness", "text_fidelity"):
                    st = paired_stats([(bj[i].get(axis), cj[i].get(axis)) for i in jids])
                    if st:
                        L.append(f"- {axis} (n={st['n']}): Δ {st['mean_diff']:+.3f}, "
                                 f"CI95 [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] → "
                                 f"{'ЗНАЧИМО' if st['significant'] else 'шум'}")
                # подмены/прожёванные — суммарно (не парно, но с n)
                sb = sum(len(bj[i].get("phoneme_substitutions") or []) for i in jids)
                sc = sum(len(cj[i].get("phoneme_substitutions") or []) for i in jids)
                mb = sum(len(bj[i].get("words_mangled") or []) for i in jids)
                mc = sum(len(cj[i].get("words_mangled") or []) for i in jids)
                L.append(f"- фонемные подмены: {sb} → {sc}; прожёванные слова: {mb} → {mc} "
                         f"(счётчики на тех же {len(jids)} файлах)")

        # 3) TTS WER / first_ok — парно по id
        cr = report_items(cfg["report"])
        rids = sorted(set(br) & set(cr))
        rids = [i for i in rids if br[i].get("kind") == "tts"]
        if rids:
            w = paired_stats([(br[i].get("wer"), cr[i].get("wer")) for i in rids])
            if w:
                L.append(f"- TTS WER (n={w['n']}): Δ {w['mean_diff']:+.4f}, "
                         f"CI95 [{w['ci95'][0]:+.4f}, {w['ci95'][1]:+.4f}] → "
                         f"{'ЗНАЧИМО' if w['significant'] else 'в пределах шума'}")
            f1 = paired_stats([(1 if br[i].get("first_ok") else 0, 1 if cr[i].get("first_ok") else 0)
                               for i in rids])
            mc = mcnemar([(bool(br[i].get("first_ok")), bool(cr[i].get("first_ok"))) for i in rids])
            L.append(f"- TTS first_ok (n={len(rids)}): McNemar b={mc['b']} c={mc['c']} p={mc['p']} → "
                     f"{'ЗНАЧИМО' if mc['significant'] else 'в пределах шума'}")

        # 4) эмо-годен — McNemar
        ce = emo_map(cfg["emo"])
        eids = sorted(set(be) & set(ce))
        if eids:
            mc = mcnemar([(be[i] == "годен", ce[i] == "годен") for i in eids])
            nb = sum(1 for i in eids if be[i] == "годен")
            nc = sum(1 for i in eids if ce[i] == "годен")
            L.append(f"- эмо годен (n={len(eids)}): {nb} → {nc} "
                     f"({nb/len(eids)*100:.1f}% → {nc/len(eids)*100:.1f}%); McNemar b={mc['b']} "
                     f"c={mc['c']} p={mc['p']} → {'ЗНАЧИМО' if mc['significant'] else 'в пределах шума'}")
        L.append("")

    L += ["## Вывод для методологии гейтов", "",
          "1. Различия, не прошедшие парный тест, НЕ должны трактоваться как регресс/улучшение.",
          "2. Для долей (first_ok, эмо-годен) при n=48/60 порог значимости ≈ 10–15 п.п.;",
          "   наши прошлые вердикты «drift» по эмоциям (48.3→43.3) в этот диапазон попадают как шум.",
          "3. Увеличить мощность: n≥120 для долей (или 3 сида на элемент), парные сравнения обязательны.",
          "4. Для осей судьи (accent/palatalization, n=33) парный CI обычно узкий — это надёжный сигнал.",
          "5. Три независимые оси произношения (решение 21.09): CER + фонемные подмены/артикуляция",
          "   + human mumble rate. Ни одна по отдельности не является KPI.", ""]

    open(os.path.join(D, "GATE_STATISTICS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(D, "GATE_STATISTICS.md"))
    for line in L:
        if line.startswith("- ") or line.startswith("## "):
            print(line)


if __name__ == "__main__":
    main()
