"""Сводка пилота s2: фактический объём данных, время обучения, авто-метрики по 5 вариантам,
фонетический пак (16 заданий). Пишет reports\\s2_compare\\COMPARISON.md + compare.json.

ВАЖНО: WER/ASR — только содержание; произношение/естественность/голос — из слепого
прослушивания (blind_s2\\listen_form.csv), слышимые дефекты приоритетнее высокого ASR.
"""
import json
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
LT = os.path.join(AUK, "local_tests")
REPORTS = os.path.join(LOCAL, "reports")
OUT = os.path.join(REPORTS, "s2_compare")

VARIANTS = [
    ("u10000", "чистый u10000", os.path.join(REPORTS, "u0_baseline", "baseline_report.json"),
     os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "phon_u10000")),
    ("A@250", "пилот A, u250", os.path.join(REPORTS, "s2_A_250", "baseline_report.json"),
     os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "phon_A_250")),
    ("A@500", "пилот A, u500", os.path.join(REPORTS, "s2_A_500", "baseline_report.json"),
     os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "phon_A_500")),
    ("B@250", "пилот B, u250", os.path.join(REPORTS, "s2_B_250", "baseline_report.json"),
     os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "phon_B_250")),
    ("B@500", "пилот B, u500", os.path.join(REPORTS, "s2_B_500", "baseline_report.json"),
     os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "phon_B_500")),
]


def train_facts():
    facts = {}
    for v, run in (("A", "run_s2_A"), ("B", "run_s2_B")):
        p = os.path.join(LOCAL, run, "accounting.json")
        if os.path.exists(p):
            a = json.load(open(p, encoding="utf-8"))
            facts[v] = {
                "updates": a.get("updates"),
                "speech_n": a.get("speech_n"), "tools_n": a.get("tools_n"),
                "speech_h_tgt": round(a.get("speech_tgt_s", 0) / 3600, 2),
                "tools_h_tgt": round(a.get("tools_tgt_s", 0) / 3600, 2),
                "speech_h_ref": round(a.get("speech_ref_s", 0) / 3600, 2),
                "tools_h_ref": round(a.get("tools_ref_s", 0) / 3600, 2),
            }
        else:
            facts[v] = None
    wall = {}
    try:
        lines = [l for l in open(os.path.join("G:\\AI\\_tmp", "s2_chain.log"), encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
    except Exception:
        lines = []
    for phase in ("A", "B"):
        t0 = t1 = None
        for l in lines:
            if re.search(rf"phase {phase}: training", l):
                t0 = l[:19]
            if re.search(rf"phase {phase} training done", l):
                t1 = l[:19]
        if t0 and t1:
            wall[phase] = f"{t0} -> {t1}"
    return facts, wall


def phon_metrics(pack_path, out_dir):
    if not os.path.exists(pack_path) or not os.path.isdir(out_dir):
        return None
    import sys
    sys.path.insert(0, LOCAL)
    from ru_metrics import transcribe_path, text_metrics

    pack = json.load(open(pack_path, encoding="utf-8"))
    wers, sims, n = [], [], 0
    for it in pack:
        wav = os.path.join(out_dir, f"{it['id']}.wav")
        if not os.path.exists(wav):
            continue
        heard, err = transcribe_path(wav)
        tm = text_metrics(it["text"], heard)
        wers.append(tm["wer"])
        n += 1
    return {"n": n, "wer_mean": round(float(np.mean(wers)), 3) if wers else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    facts, wall = train_facts()

    rows = []
    for vid, label, rep_path, ppath, pdir in VARIANTS:
        rep = json.load(open(rep_path, encoding="utf-8")) if os.path.exists(rep_path) else None
        agg = rep.get("aggregate", {}) if rep else {}
        leaks = sum(1 for it in (rep.get("items", []) if rep else [])
                    if "REF_LEAK?" in it.get("flags", []))
        phon = phon_metrics(ppath, pdir)
        rows.append({"id": vid, "label": label,
                     "tts_ok": agg.get("tts", {}).get("first_ok"),
                     "clone_ok": agg.get("clone", {}).get("first_ok"),
                     "tool_ok": agg.get("tool", {}).get("first_ok"),
                     "wer_tts": agg.get("tts", {}).get("wer_mean"),
                     "wer_clone": agg.get("clone", {}).get("wer_mean"),
                     "wer_tool": agg.get("tool", {}).get("wer_mean"),
                     "sim_tts": agg.get("tts", {}).get("sim_median"),
                     "sim_clone": agg.get("clone", {}).get("sim_median"),
                     "leaks": leaks, "phon": phon})

    L = []
    L.append("# Пилот s2 — сравнение вариантов (авто-часть)")
    L.append("")
    L.append(f"Дата: {datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    L.append("**Внимание:** WER/ASR отражает только содержание. Фонетические замены (ж/з, ч/ц, ы/и) "
             "и естественность автоматика не измеряет — решает слепое прослушивание "
             "(`local_tests\\blind_s2\\`). Высокий ASR НЕ перекрывает слышимые дефекты.")
    L.append("")
    L.append("## Фактический объём (предъявленные данные)")
    L.append("")
    L.append("| Пилот | Обновления | Речь: примеров | Речь: ч (цель) | Речь: ч (ref) | Tools: примеров | Tools: ч (цель) | Время (wall) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in ("A", "B"):
        f = facts.get(v)
        if f:
            L.append(f"| {v} | {f['updates']} | {f['speech_n']} | {f['speech_h_tgt']} | "
                     f"{f['speech_h_ref']} | {f['tools_n']} | {f['tools_h_tgt']} | {wall.get(v, '?')} |")
        else:
            L.append(f"| {v} | — | — | — | — | — | — | — |")
    L.append("")
    L.append("Небольшой объём (≈680 примеров на пилот) показывает раннюю реакцию адаптера "
             "и регрессии, но не предел качества s2.")
    L.append("")
    L.append("## Авто-метрики (одинаковые 120 заданий)")
    L.append("")
    L.append("| Вариант | TTS first_ok | Clone first_ok | Tool first_ok | WER TTS | WER Clone | WER Tool | sim TTS (med) | sim Clone (med) | Утечек рефа | Фон. WER (16) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['label']} | {r['tts_ok']} | {r['clone_ok']} | {r['tool_ok']} | "
                 f"{r['wer_tts']} | {r['wer_clone']} | {r['wer_tool']} | {r['sim_tts']} | "
                 f"{r['sim_clone']} | {r['leaks']} | {(r['phon'] or {}).get('wer_mean')} |")
    L.append("")
    L.append("## Слепое прослушивание (заполняет пользователь)")
    L.append("")
    L.append("Набор: `local_tests\\blind_s2\\` — 5 вариантов скрыты; форма: `listen_form.csv` "
             "(text_accuracy, pronunciation, naturalness, voice_similarity по каждому файлу).")
    L.append("После заполнения открыть `SECRET_map.csv` и свести оценки по вариантам.")
    L.append("")
    L.append("| Вариант | Точность текста | Произношение | Естественность | Сходство голоса |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['label']} | — | — | — | — |")
    L.append("")
    L.append("## Ключевые вопросы пилота")
    L.append("1. Становится ли русский стабильнее (меньше сырых срывов/утечек) на A vs u10000?")
    L.append("2. Сохраняет ли B инструменты лучше A — БЕЗ потери качества речи и клонирования?")
    L.append("3. Если улучшения нет — остаёмся на проверенном s1 (продолжать ради шагов не нужно).")

    md = "\n".join(L)
    open(os.path.join(OUT, "COMPARISON.md"), "w", encoding="utf-8").write(md)
    json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
               "train_facts": facts, "wall": wall, "rows": rows},
              open(os.path.join(OUT, "compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(md)
    print("COMPARE_BUILD_DONE")


if __name__ == "__main__":
    main()
