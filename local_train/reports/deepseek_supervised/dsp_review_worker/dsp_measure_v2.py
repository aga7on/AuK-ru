"""Run the corrected DSP measurement over the actual 175 s2 tool WAVs.

Inputs (read-only):
  local_tests/u0_control/manifest.json          -> 35 tool task refs + instructions
  local_tests/{u0_control,s2_A_250,s2_A_500,s2_B_250,s2_B_500}/tool_*.wav
  local_train/data_s2_tools_v3/noise_add_val_*.wav (clean ground truth provenance)

Outputs (this directory only):
  dsp_v2_perfile.json   one record per (op, task, variant), 5 variants x 5 tasks x 7 ops
  dsp_v2_summary.json   aggregate per op/variant + controls reference
  dsp_v2_inputs.json    source/output/clean paths with full sha256
  DSP_REVIEW_V2.md      concise human-readable review

No training, no synthesis, no network. Prospectively declared thresholds in
dsp_core_v2.CFG; no headline "pass" is derived from a mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsp_core_v2 as C  # noqa: E402
import dsp_controls_v2 as CTL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
OUTDIR = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
VARIANTS = ["u10000", "A@250", "A@500", "B@250", "B@500"]
DIRS = {"u10000": "u0_control", "A@250": "s2_A_250", "A@500": "s2_A_500",
        "B@250": "s2_B_250", "B@500": "s2_B_500"}
METRIC = {"volume_up": "delta_db", "volume_down": "delta_db",
          "speed_up": "rate_proxy", "speed_down": "rate_proxy",
          "pitch_up": "semitones", "pitch_down": "semitones",
          "noise_add": "snr_gain_db"}
UNIT = {"volume_up": "dB", "volume_down": "dB", "speed_up": "x", "speed_down": "x",
        "pitch_up": "st", "pitch_down": "st", "noise_add": "dB"}


def build_tasks():
    man = json.load(open(os.path.join(LT, "u0_control", "manifest.json"), encoding="utf-8"))
    refs = {m["id"]: m for m in man if m["kind"] == "tool"}
    tasks = []
    for op in C.OPS:
        for i in range(5):
            tid = f"tool_{op}_{i}"
            m = refs.get(tid)
            tasks.append({"op": op, "index": i, "task_id": tid,
                          "instruction": m.get("instruction") if m else None,
                          "ref": m.get("ref") if m else None})
    return tasks


def run(tasks, content_evidence=None):
    perfile = []
    inputs = []
    for t in tasks:
        for v in VARIANTS:
            out_path = os.path.join(LT, DIRS[v], t["task_id"] + ".wav")
            ce = None
            if content_evidence:
                ce = content_evidence.get(f"{v}/{t['task_id']}")
            rec = C.measure_file(t["op"], t["ref"], out_path, content_evidence=ce)
            rec.update(task_id=t["task_id"], variant=v, index=t["index"],
                       instruction=t["instruction"])
            perfile.append(rec)
            entry = {"variant": v, "task_id": t["task_id"], "op": t["op"],
                     "source_path": t["ref"], "output_path": out_path}
            for k in ("sha256_source", "sha256_output"):
                entry[k] = rec.get(k)
            if rec.get("clean_provenance"):
                entry["clean_provenance"] = rec["clean_provenance"]
                entry["clean_sha256"] = rec.get("clean_sha256")
            inputs.append(entry)
    return perfile, inputs


def _effects(rows, metric):
    return [r[metric] for r in rows if r.get("valid") and isinstance(r.get(metric), (int, float))]


def _worst(op, rows):
    valid = [r for r in rows if r.get("valid") and isinstance(r.get(METRIC[op]), (int, float))]
    if not valid:
        cand = [r for r in rows if r.get("align", {}).get("corr") is not None]
        if cand:
            w = min(cand, key=lambda r: r["align"]["corr"])
            return {"task_id": w["task_id"], "variant": w["variant"],
                    "metric": "align_env_corr", "value": round(float(w["align"]["corr"]), 3),
                    "note": "no applicable measurement; lowest alignment correlation"}
        return None
    if op in C.POSITIVE:
        w = min(valid, key=lambda r: r[METRIC[op]])
    elif op in C.NEGATIVE:
        w = max(valid, key=lambda r: r[METRIC[op]])
    else:  # noise_add: lowest snr_gain
        w = min(valid, key=lambda r: r[METRIC[op]])
    return {"task_id": w["task_id"], "variant": w["variant"],
            "metric": METRIC[op], "value": round(float(w[METRIC[op]]), 3)}


def summarize(perfile, controls):
    summary = {"thresholds": {k: v for k, v in C.CFG.items()}, "ops": {}, "controls": controls}
    for op in C.OPS:
        rows = [r for r in perfile if r["op"] == op]
        summary["ops"][op] = {}
        for v in VARIANTS:
            vr = [r for r in rows if r["variant"] == v]
            valid = [r for r in vr if r.get("valid")]
            effects = _effects(vr, METRIC[op])
            reasons = {}
            for r in vr:
                if not r.get("valid"):
                    reasons[r.get("reason") or "unknown"] = reasons.get(r.get("reason") or "unknown", 0) + 1
            block = {
                "n_total": len(vr), "n_valid": len(valid), "n_invalid": len(vr) - len(valid),
                "invalid_reasons": reasons,
                "metric": METRIC[op], "unit": UNIT[op],
                "effects": [round(e, 3) for e in effects],
                "median": round(float(np.median(effects)), 3) if effects else None,
                "q1": round(float(np.percentile(effects, 25)), 3) if effects else None,
                "q3": round(float(np.percentile(effects, 75)), 3) if effects else None,
                "min": round(float(np.min(effects)), 3) if effects else None,
                "max": round(float(np.max(effects)), 3) if effects else None,
                "direction_ok_n": int(sum(1 for r in valid if r.get("direction_ok"))),
                "within_window_n": int(sum(1 for r in valid if r.get("within_prospective_window"))),
                "worst": _worst(op, vr),
            }
            if op == "noise_add":
                block["applicable_n"] = int(sum(1 for r in vr if r.get("applicable")))
                block["denoise_evidence"] = sorted({r.get("denoise_evidence") for r in vr
                                                    if r.get("denoise_evidence")})
                block["noise_floor_reduction"] = [round(r["noise_floor_reduction_db"], 2)
                                                  for r in valid
                                                  if isinstance(r.get("noise_floor_reduction_db"), (int, float))]
            if op in ("volume_up", "volume_down"):
                block["clipped_output_n"] = int(sum(1 for r in vr if r.get("clipped_output")))
            summary["ops"][op][v] = block
    return summary


def md_report(tasks, perfile, summary, controls):
    L = []
    L.append("# DSP review v2 — независимый перезамер инструментов s2")
    L.append("")
    L.append("Worker: DeepSeek v4.1 Flash (DSP correction), bounded run. Источник: "
             "`dsp_core_v2.py` / `dsp_measure_v2.py`. Файлы-артефакты: 175 WAV "
             "(7 операций × 5 задач × 5 вариантов). Сырые записи: `dsp_v2_perfile.json`, "
             "агрегат: `dsp_v2_summary.json`, входные пути+sha256: `dsp_v2_inputs.json`.")
    L.append("")
    L.append("## Что исправлено относительно `tool_dsp_measure.py`")
    L.append("")
    L.append("1. **Активность:** абсолютный энергетический порог (dBFS) + порог относительно "
             "p95; all-zero/non-finite/too-short → жёстко invalid (тишина больше не «активна»). "
             "Абсолютный уровень активной речи (`active_rms_db_in/out`) пишется в per-file JSON, "
             "near-silent выходы (< −55 dBFS) флагуются — это вскрыло `tool_pitch_down_0` "
             "(все варианты ≈−66 dBFS), который старый скрипт считал валидным.")
    L.append("2. **Denoise:** сегментный SNR/корреляции только после явного выравнивания "
             "(ограниченный лаг ±300 мс) и проверки применимости по корреляции огибающих. "
             "При фазовом/временном расхождении — `inconclusive`, а не число. Добавлена "
             "фазо-устойчивая вторичная метрика (снижение шумового пола в нерабочих кадрах "
             "чистого эталона). Тишина не считается идеальным denoise. SR-мисматч — явный "
             "resample с флагом, без молчаливого обрезания.")
    L.append("3. **Громкость:** FLOAT-WAV с запасом, медиана/IQR покадрового гейна, peak и "
             "clip ratio; контроли с положительными И отрицательными ожиданиями.")
    L.append("4. **Скорость:** прокси по активному **span** + явный гейт сопоставимости "
             "контента (stretch-инвариантный MFCC nearest-neighbour + DTW); если контент "
             "несопоставим — invalid. **Питч:** парные voiced-кадры (а не голое отношение "
             "медиан), voiced coverage, octave-guard.")
    L.append("5. **Чистый эталон** берётся из происхождения датасета "
             "(`noise_add_val_<id>.wav`, проверено равенство kyutai-клипу), пути и sha256 "
             "логируются.")
    L.append("6. **Агрегация:** n/valid/invalid по op×variant, распределение эффекта, "
             "перспективные пороги; среднее НЕ объявляется вердиктом.")
    L.append("")
    n_ok = sum(1 for r in controls.values() if r["asserted_ok"])
    L.append(f"## Контроли пайплайна: {n_ok}/{len(controls)} asserted")
    L.append("")
    L.append("| контроль | ожидание | asserted |")
    L.append("|---|---|---|")
    for name, r in controls.items():
        L.append(f"| {name} | {r['expectation']} | {'PASS' if r['asserted_ok'] else 'FAIL'} |")
    L.append("")
    L.append("Полные наблюдения контролей: `dsp_v2_controls.json`.")
    L.append("")
    L.append("## Перспективные пороги (объявлены до прогона, не подогнаны)")
    L.append("")
    L.append(f"- volume: up {C.CFG['th_volume_up_db']} dB (цель +6), down {C.CFG['th_volume_down_db']} dB (цель −6)")
    L.append(f"- speed: up {C.CFG['th_speed_up']} (цель 1.1), down {C.CFG['th_speed_down']} (цель 0.9)")
    L.append(f"- pitch: up {C.CFG['th_pitch_up_st']} st (цель +2), down {C.CFG['th_pitch_down_st']} st (цель −2)")
    L.append(f"- denoise: применимо при corr огибающих ≥ {C.CFG['min_env_corr']} и |лаг| ≤ {C.CFG['max_lag_ms']} мс; "
             f"вторично снижение шумового пола ≥ {C.CFG['th_noise_floor_reduction_db']} dB")
    L.append("")
    L.append("## Результаты по операциям (знаменатель n=5 на вариант)")
    L.append("")
    for op in C.OPS:
        L.append(f"### {op} — метрика `{METRIC[op]}` ({UNIT[op]})")
        L.append("")
        if op == "noise_add":
            L.append("| вариант | valid/5 | applicable | snr_gain_db median [min..max] | noise_floor_red dB | invalid | worst |")
            L.append("|---|---:|---:|---|---:|---|---|")
        elif op in ("volume_up", "volume_down"):
            L.append("| вариант | valid/5 | delta_db median [min..max] | dir ok/valid | in window | clip out | invalid | worst |")
            L.append("|---|---:|---|---:|---:|---:|---|---|")
        else:
            L.append(f"| вариант | valid/5 | {METRIC[op]} median [min..max] | dir ok/valid | in window | invalid | worst |")
            L.append("|---|---:|---|---:|---:|---|---|")
        for v in VARIANTS:
            b = summary["ops"][op][v]
            inv = ", ".join(f"{k}×{c}" for k, c in b["invalid_reasons"].items()) or "—"
            w = b["worst"]
            ws = f"{w['task_id']}@{w['variant']}={w['value']}" if w else "—"
            rng = f"[{b['min']}..{b['max']}]" if b["median"] is not None else "—"
            med = f"{b['median']} {rng}" if b["median"] is not None else "—"
            if op == "noise_add":
                nfr = b.get("noise_floor_reduction")
                nfr_s = f"{np.median(nfr):.2f}" if nfr else "—"
                L.append(f"| {v} | {b['n_valid']}/{b['n_total']} | {b.get('applicable_n', 0)} | {med} | "
                         f"{nfr_s} | {inv} | {ws} |")
            elif op in ("volume_up", "volume_down"):
                L.append(f"| {v} | {b['n_valid']}/{b['n_total']} | {med} | {b['direction_ok_n']}/{b['n_valid']} | "
                         f"{b['within_window_n']}/{b['n_valid']} | {b.get('clipped_output_n', 0)} | {inv} | {ws} |")
            else:
                L.append(f"| {v} | {b['n_valid']}/{b['n_total']} | {med} | {b['direction_ok_n']}/{b['n_valid']} | "
                         f"{b['within_window_n']}/{b['n_valid']} | {inv} | {ws} |")
        L.append("")
    L.append("## Наблюдения (факты, без вердикта по вариантам)")
    L.append("")
    O = summary["ops"]
    vu, vd = O["volume_up"], O["volume_down"]
    L.append(f"- **volume_up:** запрошено +6 dB, но медиана по вариантам {[vu[v]['median'] for v in VARIANTS]} dB "
             f"(в окно {C.CFG['th_volume_up_db']} попало {[vu[v]['within_window_n'] for v in VARIANTS]} из 5). "
             f"Направление верное, величина сильно недобрана.")
    wvd = vd['B@500']['worst']
    L.append(f"- **volume_down:** запрошено −6 dB, медиана {[vd[v]['median'] for v in VARIANTS]} dB, "
             f"верное направление лишь {[vd[v]['direction_ok_n'] for v in VARIANTS]}/5. "
             f"Худший (B@500): {wvd['task_id']}={wvd['value']} dB (рост, а не снижение).")
    L.append(f"- **speed_down:** сопоставимый контент есть только у B@500 "
             f"({[O['speed_down'][v]['n_valid'] for v in VARIANTS]}/5 valid); у остальных прокси ненадёжен → invalid, "
             f"а не ложный «rate».")
    L.append(f"- **speed_up:** B@500 — единственный с 5/5 valid; прочие имеют 0-2 несопоставимых.")
    L.append(f"- **pitch_up:** стабильно ~+2 st во всех вариантах (парные voiced-кадры).")
    L.append(f"- **pitch_down:** 3/5 valid; `tool_pitch_down_0` у всех вариантов near-silent "
             f"(≈−66 dBFS), `tool_pitch_down_3` — источник почти без voiced-кадров (coverage 0.14) → invalid.")
    nfr_med = []
    for v in VARIANTS:
        vals = O['noise_add'][v].get('noise_floor_reduction') or []
        nfr_med.append(round(float(np.median(vals)), 2) if vals else None)
    L.append(f"- **noise_add:** применимость только {[O['noise_add'][v].get('applicable_n') for v in VARIANTS]}/5; "
             f"где применимо — сегментный SNR-gain отрицательный (метрика штрафует ресинтез), "
             f"но фазо-устойчивое снижение шумового пола (медиана, dB) {nfr_med} > 0. "
             f"Вывод о denoise не делается (inconclusive/weak); худший применимый — "
             f"{O['noise_add']['B@500']['worst']['task_id']}={O['noise_add']['B@500']['worst']['value']} dB.")
    L.append("")
    L.append("## Denoise — применимость и ограничения")
    L.append("")
    L.append("Сегментный SNR считается против чистого клипа. Для регенерированного нейро-аудио "
             "это слабая метрика: любое расхождение фазы/просодии/синтеза штрафуется как «шум». "
             "Поэтому замер помечается применимым только при хорошем выравнивании, иначе — "
             "`inconclusive`. Ни один случай не объявляется доказанным denoise.")
    L.append("")
    for v in VARIANTS:
        b = summary["ops"]["noise_add"][v]
        L.append(f"- {v}: applicable {b.get('applicable_n', 0)}/5, evidence={b.get('denoise_evidence')}")
    L.append("")
    L.append("## Контент-доказательства (ASR/Gemini)")
    L.append("")
    L.append("Проверенного per-file mapping варианта на строки автосудьи в репозитории нет без "
             "открытия `local_tests/blind_s2/SECRET_map.csv`, что запрещено до завершения "
             "слепого прослушивания. Поэтому content-evidence не подставлялся. Доступен только "
             "агрегат `exploratory/blind_s2_gemini_v2_by_variant.md` (вспомогательный, не решение). "
             "Скрипт поддерживает `--content-evidence PATH` для привязки после появления "
             "верифицированного mapping.")
    L.append("")
    L.append("## Оговорки")
    L.append("")
    L.append("- Никакой агрегатный «pass» не объявляется: решение по вариантам требует слепого "
             "прослушивания; этот отчёт — объективный magnitude-замер.")
    L.append("- speed — прокси активного span, условный на сопоставимость контента; pitch — "
             "парные voiced-кадры; оба помечаются invalid при несопоставимости.")
    L.append("- Один фиксированный seed librosa/pyin; артефакты не перегенерировались.")
    L.append("")
    L.append(f"Всего записей per-file: {len(perfile)}; контролей: {len(controls)} "
             f"({n_ok} asserted).")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content-evidence", default=None,
                    help="optional JSON mapping 'variant/task_id' -> evidence dict")
    args = ap.parse_args()
    ce = json.load(open(args.content_evidence, encoding="utf-8")) if args.content_evidence else None
    tasks = build_tasks()
    perfile, inputs = run(tasks, ce)
    controls = CTL.run_controls()
    summary = summarize(perfile, controls)
    json.dump(perfile, open(os.path.join(HERE, "dsp_v2_perfile.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(inputs, open(os.path.join(HERE, "dsp_v2_inputs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(summary, open(os.path.join(HERE, "dsp_v2_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    md = md_report(tasks, perfile, summary, controls)
    open(os.path.join(HERE, "DSP_REVIEW_V2.md"), "w", encoding="utf-8").write(md + "\n")
    n_ok = sum(1 for r in controls.values() if r["asserted_ok"])
    print(f"wrote DSP_REVIEW_V2.md, dsp_v2_perfile.json, dsp_v2_summary.json, dsp_v2_inputs.json")
    print(f"perfile={len(perfile)} controls={n_ok}/{len(controls)}")
    for op in C.OPS:
        b = summary["ops"][op]["B@500"]
        print(f"  {op:11s} B@500 valid={b['n_valid']}/{b['n_total']} "
              f"median={b['median']} dir_ok={b['direction_ok_n']} worst={b['worst']}")


if __name__ == "__main__":
    main()
