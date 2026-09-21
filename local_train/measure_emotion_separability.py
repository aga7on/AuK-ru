# -*- coding: utf-8 -*-
"""Объективная проверка: различимы ли эмоции АКУСТИЧЕСКИ (не по судье).

Мотив: в model card заявлено «эмоции работают: 48.3% годен по автосудье». Но автосудья
согласуется с человеком лишь на 21%, а Aniemore SER невалиден на синтетике (disgust-collapse).
Поэтому нужен независимый акустический тест: если эмоции реально передаются, группы
(happy/sad/angry/fearful/excited) должны различаться по f0/energy/tempo.

Метрики на файл: f0 median, f0 range, f0 std, RMS, RMS динамика (p90/p10), speech rate
(доля voiced-фреймов), spectral centroid (яркость).

Тест разделимости:
  1. однофакторный ANOVA-подобный критерий: доля между-групповой дисперсии (η²) по каждой
     метрике — если η² мала (<0.1), эмоции НЕ различимы по этой метрике;
  2. leave-one-out классификация по 5 эмоциям (ближайший центроид, нормированные метрики) —
     accuracy против случайных 20%;
  3. Cohen's d для пары happy vs sad (самая ожидаемая контрастная пара).

usage: python measure_emotion_separability.py [--dir local_tests/emotion_ru_s7_5750]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"G:\AI\AuK\src")


def feats(x, sr):
    import numpy as np
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    f0s, rms_f = [], []
    for i in range(0, max(1, len(x) - frame), hop):
        seg = x[i:i + frame]
        r = float(np.sqrt(np.mean(seg ** 2)))
        rms_f.append(r)
        if r < 0.01:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        lo, hi = int(sr / 350), int(sr / 70)
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            f0s.append(sr / lag)
    if not f0s:
        return None
    f0s = np.array(f0s)
    rms_f = np.array(rms_f) + 1e-9
    # spectral centroid
    win = min(len(x), sr)
    spec = np.abs(np.fft.rfft(x[:win] * np.hanning(win)))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-9))
    return {
        "f0_med": float(np.median(f0s)), "f0_range": float(f0s.max() - f0s.min()),
        "f0_std": float(f0s.std()), "rms": float(np.sqrt(np.mean(x ** 2))),
        "rms_dyn": float(np.percentile(rms_f, 90) / np.percentile(rms_f, 10)),
        "voiced_ratio": len(f0s) / max(1, len(rms_f)),
        "centroid": centroid,
    }


def eta_squared(groups):
    """η² = between-group SS / total SS для dict emotion -> list of values."""
    import statistics as st
    allv = [v for vs in groups.values() for v in vs]
    if len(allv) < 3 or len(groups) < 2:
        return None
    gm = st.mean(allv)
    ss_tot = sum((v - gm) ** 2 for v in allv)
    ss_bet = sum(len(vs) * (st.mean(vs) - gm) ** 2 for vs in groups.values() if vs)
    return round(ss_bet / ss_tot, 3) if ss_tot else None


def cohens_d(a, b):
    import statistics as st
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return None
    return round((ma - mb) / pooled ** 0.5, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"G:\AI\AuK\local_tests\emotion_ru_s7_5750")
    ap.add_argument("--out", default=r"G:\AI\AuK\local_train\reports\deepseek_supervised\EMOTION_SEPARABILITY.md")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    meta = json.load(open(os.path.join(args.dir, "results.json"), encoding="utf-8"))
    rows = []
    for m in meta:
        if m.get("status") != "ok" or not os.path.exists(m["file"]):
            continue
        x, sr = sf.read(m["file"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f = feats(x, sr)
        if not f:
            continue
        rows.append({**f, "emotion": m["emotion"], "voice": m.get("voice"), "seed": m.get("seed")})

    print("analysed files:", len(rows))
    by_emo = defaultdict(list)
    for r in rows:
        by_emo[r["emotion"]].append(r)
    print("emotions:", {k: len(v) for k, v in by_emo.items()})

    METRICS = ["f0_med", "f0_range", "f0_std", "rms", "rms_dyn", "voiced_ratio", "centroid"]
    # η² по всем данным и внутри одного голоса (чтобы убрать конфунд пола)
    lines = ["# EMOTION SEPARABILITY — различимы ли эмоции акустически", "",
             f"Источник: `{os.path.basename(args.dir)}` (n={len(rows)} файлов, "
             f"{len(by_emo)} эмоций). Метод: независимые акустические замеры, БЕЗ автосудьи",
             "(автосудья согласуется с человеком лишь на 21%; Aniemore SER невалиден на синтетике).", "",
             "## η² — доля между-групповой дисперсии (0 = эмоции неразличимы, 1 = полностью)", "",
             "| метрика | η² (все) | η² (внутри голоса) | интерпретация |", "|---|---|---|---|"]
    weak_all = 0
    for met in METRICS:
        g_all = {e: [r[met] for r in rs] for e, rs in by_emo.items()}
        e_all = eta_squared(g_all)
        # внутри голоса: усредняем η² по голосам
        per_voice = []
        for voice in set(r["voice"] for r in rows):
            gv = defaultdict(list)
            for r in rows:
                if r["voice"] == voice:
                    gv[r["emotion"]].append(r[met])
            ev = eta_squared(gv)
            if ev is not None:
                per_voice.append(ev)
        e_v = round(sum(per_voice) / len(per_voice), 3) if per_voice else None
        interp = ("слабо" if (e_all or 0) < 0.10 else ("умеренно" if (e_all or 0) < 0.25 else "сильно"))
        if (e_all or 0) < 0.10:
            weak_all += 1
        lines.append(f"| {met} | {e_all} | {e_v} | {interp} |")

    # LOO nearest-centroid accuracy ВНУТРИ голоса.
    # БАГ-ФИКС (21.09): первая версия пулила центроиды по всем файлам, смешивая мужской и
    # женский голоса. Пол давал f0_med ~150 vs ~220 Гц — расстояние определялось полом,
    # а не эмоцией, и дало accuracy 0.0/60 (абсурд для 5 классов: случайно ~12).
    # Правильный дизайн: классифицировать файл только против пула ТОГО ЖЕ голоса.
    import statistics as st
    emos = sorted(by_emo)
    MET = METRICS

    def loo_accuracy(data, within_voice=True):
        correct = 0
        for i, r in enumerate(data):
            best, best_d = None, None
            for e in emos:
                pool = [x for j, x in enumerate(data)
                        if j != i and x["emotion"] == e
                        and (not within_voice or x["voice"] == r["voice"])]
                if len(pool) < 2:
                    continue
                d = 0.0
                for m in MET:
                    sd = (st.pstdev([p[m] for p in data]) or 1e-9)
                    d += ((r[m] - st.mean([p[m] for p in pool])) / sd) ** 2
                d = d ** 0.5
                if best_d is None or d < best_d:
                    best_d, best = d, e
            correct += (best == r["emotion"])
        return round(correct / len(data), 3) if data else None

    acc = loo_accuracy(rows, within_voice=True)
    acc_mixed = loo_accuracy(rows, within_voice=False)

    # перестановочный тест: null-распределение accuracy при случайных метках эмоции
    import random
    rng = random.Random(7)
    null = []
    for _ in range(200):
        sh = [dict(r) for r in rows]
        labels = [r["emotion"] for r in rows]
        rng.shuffle(labels)
        for r, lab in zip(sh, labels):
            r["emotion"] = lab
        null.append(loo_accuracy(sh, within_voice=True))
    null.sort()
    p_perm = round(sum(1 for v in null if v >= (acc or 0)) / len(null), 3)
    lines += ["", "## Leave-one-out классификация эмоции по акустике", "",
              f"Accuracy (ВНУТРИ голоса, корректный дизайн): **{acc}**",
              f"Accuracy (смешанные голоса — некорректно, конфунд пола): {acc_mixed}",
              f"Случайный уровень: {round(1/len(emos),3)} (5 эмоций)",
              f"Перестановочный тест (200 шафлов): null median {null[len(null)//2]}, "
              f"95-й процентиль {null[int(0.95*len(null))]}, **p={p_perm}**", "",
              f"→ {'сигнала НЕТ (p>0.05, accuracy на уровне случайного)' if (p_perm or 1) > 0.05 else 'есть сигнал (p≤0.05)'}",
              "",
              "Примечание: дизайн сбалансирован — 2 голоса × 5 эмоций × 6 файлов = 60,",
              "поэтому пол является конфундом и обязан контролироваться (within-voice).", ""]

    # Cohen's d happy vs sad
    if "happy" in by_emo and "sad" in by_emo:
        lines += ["## Cohen's d: happy vs sad (ожидаемо самая контрастная пара)", "",
                  "| метрика | d | интерпретация |", "|---|---|---|"]
        for met in METRICS:
            a = [r[met] for r in by_emo["happy"]]
            b = [r[met] for r in by_emo["sad"]]
            d = cohens_d(a, b)
            mag = ("нет эффекта" if d is None or abs(d) < 0.2 else
                   ("малый" if abs(d) < 0.5 else ("средний" if abs(d) < 0.8 else "большой")))
            lines.append(f"| {met} | {d} | {mag} |")

    lines += ["", "## Вывод", "",
              f"- η² < 0.10 (слабая связь с эмоцией) у **{weak_all}/{len(METRICS)}** метрик;",
              f"- LOO-классификация эмоции: {acc} при случайном {round(1/len(emos),3)};",
              "- интерпретация: **акустически эмоции передаются слабо или не передаются**.",
              "Это согласуется с тем, что автосудья ставил «годен», а человек эмоции не слышит",
              "(согласованность человек↔судья 21%). Заявления «эмоции работают» в релизной",
              "документации следует смягчить до «частично/не подтверждено независимой метрикой».", ""]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("η² weak metrics:", weak_all, "/", len(METRICS))
    print("LOO accuracy:", acc, "| chance:", round(1 / len(emos), 3))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
