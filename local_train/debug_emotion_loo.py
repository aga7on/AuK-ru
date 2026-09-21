# -*- coding: utf-8 -*-
"""Отладка LOO-классификатора эмоций: accuracy 0.017 при chance 0.2 математически
неправдоподобна для nearest-centroid → подозрение на баг инструмента, а не результат.

Проверяет: матрицу путаницы, accuracy по одной метрике f0_med, перестановочный null,
и главное — насколько вообще различаются файлы внутри голоса (если эмоции игнорируются,
все файлы голоса почти идентичны, и классификатор ловит шум/артефакты сида).
"""
import json
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

D = r"G:\AI\AuK\local_tests\emotion_ru_s7_5750"
METRICS = ["f0_med", "f0_range", "f0_std", "rms", "rms_dyn", "voiced_ratio", "centroid"]


def feats(x, sr):
    import numpy as np
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    f0s, rms_f = [], []
    for i in range(0, max(1, len(x) - frame), hop):
        seg = x[i:i + frame]
        r = float(np.sqrt(np.mean(seg ** 2)))
        rms_f.append(r)
        if r < 0.01:
            continue
        s = seg - seg.mean()
        ac = np.correlate(s, s, "full")[frame - 1:]
        lo, hi = int(sr / 350), int(sr / 70)
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            f0s.append(sr / lag)
    if not f0s:
        return None
    import numpy as _np
    f0s = _np.array(f0s); rms_f = _np.array(rms_f) + 1e-9
    win = min(len(x), sr)
    spec = _np.abs(_np.fft.rfft(x[:win] * _np.hanning(win)))
    freqs = _np.fft.rfftfreq(win, 1.0 / sr)
    return {"f0_med": float(_np.median(f0s)), "f0_range": float(f0s.max() - f0s.min()),
            "f0_std": float(f0s.std()), "rms": float(_np.sqrt(_np.mean(x ** 2))),
            "rms_dyn": float(_np.percentile(rms_f, 90) / _np.percentile(rms_f, 10)),
            "voiced_ratio": len(f0s) / max(1, len(rms_f)),
            "centroid": float((spec * freqs).sum() / max(spec.sum(), 1e-9))}


def main():
    import soundfile as sf
    meta = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
    rows = []
    for m in meta:
        if m.get("status") != "ok" or not os.path.exists(m["file"]):
            continue
        x, sr = sf.read(m["file"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f = feats(x, sr)
        if f:
            rows.append({**f, "emotion": m["emotion"], "voice": m["voice"], "file": os.path.basename(m["file"])})
    print("rows:", len(rows))
    emos = sorted(set(r["emotion"] for r in rows))
    voices = sorted(set(r["voice"] for r in rows))
    print("emotions:", emos, "| voices:", voices)

    # насколько файлы внутри голоса вообще различаются?
    for v in voices:
        sub = [r for r in rows if r["voice"] == v]
        f0 = [r["f0_med"] for r in sub]
        rms = [r["rms"] for r in sub]
        print(f"{v}: f0_med min {min(f0):.1f} max {max(f0):.1f} sd {st.pstdev(f0):.2f} | "
              f"rms min {min(rms):.4f} max {max(rms):.4f} sd {st.pstdev(rms):.4f}")

    # --- LOO nearest-centroid, within voice ---
    def loo(data, metrics, within=True):
        preds = []
        for i, r in enumerate(data):
            best, bd = None, None
            for e in emos:
                pool = [x for j, x in enumerate(data) if j != i and x["emotion"] == e
                        and (not within or x["voice"] == r["voice"])]
                if len(pool) < 2:
                    continue
                d = 0.0
                for m in metrics:
                    sd = st.pstdev([x[m] for x in data]) or 1e-9
                    d += ((r[m] - st.mean([p[m] for p in pool])) / sd) ** 2
                d = d ** 0.5
                if bd is None or d < bd:
                    bd, best = d, e
            preds.append((r["emotion"], best))
        acc = sum(1 for t, p in preds if t == p) / len(preds)
        return acc, preds

    acc_all, preds = loo(rows, METRICS)
    print("LOO all metrics within-voice acc: %.3f" % acc_all)
    print("confusion (true -> predicted):", Counter((t, p) for t, p in preds).most_common(8))
    print("predicted distribution:", Counter(p for _, p in preds))

    acc_f0, _ = loo(rows, ["f0_med"])
    print("LOO f0_med only acc: %.3f" % acc_f0)

    rng = random.Random(7)
    null = []
    for _ in range(200):
        sh = [dict(r) for r in rows]
        labs = [r["emotion"] for r in rows]
        rng.shuffle(labs)
        for r, l in zip(sh, labs):
            r["emotion"] = l
        null.append(loo(sh, METRICS)[0])
    null.sort()
    print("permutation null: median %.3f  p5 %.3f  p95 %.3f  max %.3f" %
          (null[len(null)//2], null[int(0.05*len(null))], null[int(0.95*len(null))], null[-1]))
    p = sum(1 for v in null if v >= acc_all) / len(null)
    print("p-value: %.3f" % p)


if __name__ == "__main__":
    main()
