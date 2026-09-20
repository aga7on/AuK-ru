# -*- coding: utf-8 -*-
"""S12: акустическая эмоциональная дистанция — объективный прокси БЕЗ SER-классификатора.

Мотивация: Aniemore невалиден на синтетике (disgust-collapse). Вместо классификации —
сопоставление СТАТИСТИК: для каждой эмоции берём teacher-распределение (dialogs_emotional)
и измеряем, насколько генерация ближе к своему teacher-профилю, чем к чужим.

Признаки (устойчивы к домену):
  f0_med, f0_range (p90−p10), energy_dyn (std RMS-кадров в dB), rate (слов/с ≈ слогов/с через ZCR-сегменты — грубо: voiced share), duration.

Метрика per emotion e: dist_e = ||feat(gen) − mean_teacher(e)|| / spread_teacher(e) (нормированная).
Integrity-проверка: argmin_e dist_e == целевая эмоция? hit-rate по набору = «acoustic emotion accuracy».
Дополнительно: intensity-монотонность (energy_dyn и f0_range должны расти low<mid<high).

usage:
  python acoustic_emo_dist.py --teacher-profile emo_teacher_profile.json --gens <results.json эмо-теста> --out report.json
  (профиль учителя строится этим же скриптом с --build-profile)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

SRC = r"G:\AI\_datasets\dialogs_emotional"
AUK = r"G:\AI\AuK"


def features(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < sr // 2:
        return None
    # f0 через автокорреляцию по кадрам (грубо, но доменно-устойчиво)
    win, hop = int(0.04 * sr), int(0.01 * sr)
    f0s = []
    for i in range(0, max(1, len(x) - win), hop):
        fr = x[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        peak = lo + int(np.argmax(ac[lo:hi]))
        if ac[peak] > 0.3 * ac[0]:
            f0s.append(sr / peak)
    f0s = np.array(f0s) if f0s else np.array([0.0])
    # energy dynamics
    frames = np.array([np.sqrt(np.mean(x[i:i + win] ** 2) + 1e-12)
                       for i in range(0, max(1, len(x) - win), hop)])
    db = 20 * np.log10(frames)
    active = db[db > db.max() - 40]
    feats = {
        "f0_med": float(np.median(f0s)),
        "f0_range": float(np.percentile(f0s, 90) - np.percentile(f0s, 10)) if len(f0s) > 4 else 0.0,
        "energy_dyn": float(np.std(active)) if len(active) > 2 else 0.0,
        "voiced_share": float(len(f0s) * hop / len(x)),
        "dur": float(len(x) / sr),
    }
    return feats


KEYS = ("f0_med", "f0_range", "energy_dyn", "voiced_share")


def build_profile(n_per_emo=120):
    import random
    lines = open(os.path.join(SRC, "downloaded_rows.csv"), encoding="utf-8").read().splitlines()
    hdr = lines[0].split("|")
    rows = [dict(zip(hdr, l.split("|"))) for l in lines[1:] if len(l.split("|")) == len(hdr)]
    rng = random.Random(5)
    by_emo = defaultdict(list)
    for r in rows:
        by_emo[r["emotion"]].append(r)
    profile = {}
    for emo, rs in by_emo.items():
        if emo in ("neutral",):
            continue
        feats = []
        for r in rng.sample(rs, min(n_per_emo, len(rs))):
            p = os.path.join(SRC, r["audio_path"])
            if os.path.exists(p):
                f = features(p)
                if f:
                    feats.append(f)
        if len(feats) < 10:
            continue
        arr = {k: np.array([f[k] for f in feats]) for k in KEYS}
        profile[emo] = {"n": len(feats),
                        "mean": {k: float(arr[k].mean()) for k in KEYS},
                        "std": {k: float(arr[k].std() + 1e-6) for k in KEYS}}
        print(f"profiled {emo}: n={len(feats)} f0_med={profile[emo]['mean']['f0_med']:.0f} "
              f"f0_range={profile[emo]['mean']['f0_range']:.0f} dyn={profile[emo]['mean']['energy_dyn']:.1f}")
    return profile


def distance(feat, prof):
    return float(np.mean([abs(feat[k] - prof["mean"][k]) / prof["std"][k] for k in KEYS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-profile", action="store_true")
    ap.add_argument("--teacher-profile", default=os.path.join(AUK, "local_train", "emo_teacher_profile.json"))
    ap.add_argument("--gens", default=None, help="results.json эмо-теста (fields: emotion,file)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.build_profile:
        prof = build_profile()
        json.dump(prof, open(args.teacher_profile, "w", encoding="utf-8"), indent=1)
        print("profile saved:", args.teacher_profile, "emotions:", list(prof))
        return

    prof = json.load(open(args.teacher_profile, encoding="utf-8"))
    MAP = {"happy": "happy", "sad": "sad", "angry": "angry", "fearful": "fear",
           "excited": "surprise", "fear": "fear", "surprise": "surprise", "disgust": "disgust"}
    rows = json.load(open(args.gens, encoding="utf-8"))
    hits = tot = 0
    per_emo = defaultdict(lambda: [0, 0])
    detail = []
    for r in rows:
        emo = (r.get("emotion") or "").lower()
        t = MAP.get(emo)
        if not t or t not in prof or not r.get("file") or not os.path.exists(r["file"]):
            continue
        f = features(r["file"])
        if not f:
            continue
        dists = {e: distance(f, p) for e, p in prof.items()}
        pred = min(dists, key=dists.get)
        ok = pred == t
        hits += ok
        tot += 1
        per_emo[emo][0] += ok
        per_emo[emo][1] += 1
        detail.append({"id": r.get("id"), "target": t, "pred": pred, "hit": ok,
                       "dist_own": round(dists[t], 3)})
    acc = hits / tot if tot else None
    print(f"ACOUSTIC_EMO acc={acc} n={tot}")
    for e, (h, n) in sorted(per_emo.items()):
        print(f"  {e}: {h}/{n}")
    if args.out:
        json.dump({"acc": acc, "n": tot, "per_emo": {k: v for k, v in per_emo.items()},
                   "rows": detail}, open(args.out, "w", encoding="utf-8"), indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
