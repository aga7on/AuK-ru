# -*- coding: utf-8 -*-
"""S12 prep: аудит whisper-клипов teacher-датасета (dialogs_emotional).

Гипотеза (S12_DESIGN): «whisper превращался в крик» потому, что часть teacher whisper-клипов
громкие/звонкие. Проверяем: RMS (dBFS), F0-median, share voiced, динамика — whisper против
остальных эмоций того же датасета.

Метод: берём из downloaded_rows.csv все whisper-клипы + случайную выборку других эмоций,
считаем статистику по фактическим wav (метаданные: audio_path относительно SRC).

usage: python whisper_audit.py [--n_per_emo 60]
Выход: local_train/reports/deepseek_supervised/WHISPER_AUDIT.md
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

SRC = r"G:\AI\_datasets\dialogs_emotional"
OUT = r"G:\AI\AuK\local_train\reports\deepseek_supervised\WHISPER_AUDIT.md"


def load_rows():
    lines = open(os.path.join(SRC, "downloaded_rows.csv"), encoding="utf-8").read().splitlines()
    hdr = lines[0].split("|")
    rows = []
    for l in lines[1:]:
        p = l.split("|")
        if len(p) == len(hdr):
            rows.append(dict(zip(hdr, p)))
    return rows


def stats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    act = x[np.abs(x) > 0.005]
    if len(act) < int(0.2 * sr):
        return None
    rms_db = float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12))
    peak_db = float(20 * np.log10(np.max(np.abs(x)) + 1e-12))
    # грубый F0 через zero-crossing (достаточно для относительного сравнения)
    zc = float(np.mean(np.abs(np.diff(np.sign(x))) > 0) * sr / 2)
    return {"rms_db": round(rms_db, 1), "peak_db": round(peak_db, 1), "zc_hz": round(zc, 0),
            "dur": round(len(x) / sr, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_emo", type=int, default=60)
    args = ap.parse_args()

    rows = load_rows()
    by_emo = defaultdict(list)
    for r in rows:
        by_emo[r["emotion"]].append(r)
    rng = random.Random(12)

    agg = {}
    for emo, rs in by_emo.items():
        sample = rng.sample(rs, min(args.n_per_emo, len(rs)))
        vals = []
        for r in sample:
            p = os.path.join(SRC, r["audio_path"])
            if not os.path.exists(p):
                continue
            s = stats(p)
            if s:
                vals.append(s)
        if not vals:
            continue
        agg[emo] = {
            "n": len(vals),
            "rms_med": float(np.median([v["rms_db"] for v in vals])),
            "rms_p75": float(np.percentile([v["rms_db"] for v in vals], 75)),
            "peak_med": float(np.median([v["peak_db"] for v in vals])),
            "zc_med": float(np.median([v["zc_hz"] for v in vals])),
        }

    L = ["# WHISPER AUDIT — teacher-датасет dialogs_emotional (S12 prep)", "",
         f"Выборка: ≤{args.n_per_emo} клипов на эмоцию. RMS/peak в dBFS по активным сегментам; zc_hz — грубый F0-прокси.", "",
         "| эмоция | n | RMS med | RMS p75 | peak med | zc med |", "|---|---|---|---|---|---|"]
    for emo, a in sorted(agg.items(), key=lambda kv: -kv[1]["rms_med"]):
        L.append(f"| {emo} | {a['n']} | {a['rms_med']:.1f} | {a['rms_p75']:.1f} | {a['peak_med']:.1f} | {a['zc_med']:.0f} |")

    w = agg.get("whisper")
    others = [a["rms_med"] for e, a in agg.items() if e != "whisper"]
    verdict = "n/a"
    if w and others:
        om = float(np.median(others))
        d = w["rms_med"] - om
        verdict = ("ГИПОТЕЗА ПОДТВЕРЖДЕНА: whisper громче прочих эмоций на %.1f дБ — фильтр по RMS обязателен" % d
                   if d > -3.0 else
                   "гипотеза не подтверждена: whisper тише прочих на %.1f дБ — крик в s5/s7 не от громкости учителя, а от формата инструкции/данных" % (-d))
    L += ["", "## Вердикт", "", verdict, "",
          "Порог: whisper должен быть ≥3 дБ ТИШЕ медианы прочих эмоций. Если нет — в S12-микс брать только",
          "клипы с RMS < (медиана спикера − 3 дБ); остальные whisper-клипы исключить."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print(verdict)


if __name__ == "__main__":
    main()
