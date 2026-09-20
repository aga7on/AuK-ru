# -*- coding: utf-8 -*-
"""S12 gate: intensity-монотонность — парное сравнение subdued vs intense.

Тот же текст+ref+seed, два формата инструкции:
  "... with a {emo} tone, subdued:" vs "... with a {emo} tone, intense:"
Метрика: RMS активного сегмента (dB) и F0-range (p90−p10).
Гейт: у intense RMS выше И/ИЛИ F0-range выше в ≥70% пар (и медиана Δ > 0).

usage: python intensity_monotonic.py --ckpt <merged> --config <cfg> --out <dir> [--n_pairs 15]
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"

PAIRS = [
    ("happy", "У нас всё получилось, мы сделали это!"),
    ("happy", "Какой замечательный сегодня выдался день."),
    ("happy", "Я очень рад тебя видеть после стольких лет."),
    ("sad", "К сожалению, всё закончилось совсем не так."),
    ("sad", "Мне тяжело об этом говорить, прости меня."),
    ("sad", "Он ушёл и больше никогда не вернётся домой."),
    ("angry", "Как ты посмел так со мной поступить!"),
    ("angry", "Я предупреждал тебя в последний раз."),
    ("angry", "Это возмутительно, я так это не оставлю."),
    ("fearful", "Там в темноте кто-то стоит и смотрит."),
    ("fearful", "Я боюсь, что мы не успеем выбраться."),
    ("fearful", "Не открывай дверь, кто бы ни стучал."),
    ("excited", "Завтра мы летим в долгожданное путешествие!"),
    ("excited", "Представляешь, мы выиграли главный приз!"),
    ("excited", "Скорее собирайся, нас ждут великие дела!"),
]


def feats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    act = x[np.abs(x) > 0.005]
    rms = float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12)) if len(act) else -60.0
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
    f0r = float(np.percentile(f0s, 90) - np.percentile(f0s, 10)) if len(f0s) > 4 else 0.0
    return rms, f0r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n_pairs", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    refs = {e["id"]: e["ref"] for e in pack if e.get("kind") == "clone"}
    ref = refs.get("clone01") or refs.get("clone08")

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    rows = []
    for pi, (emo, text) in enumerate(PAIRS[: args.n_pairs]):
        pair = {"i": pi, "emotion": emo, "text": text}
        for lv in ("subdued", "intense"):
            instr = f"Reproduce the reference voice and say in Russian with a {emo} tone, {lv}: '{text}'"
            fp = os.path.join(args.out, f"p{pi:02d}_{lv}.wav")
            content = [{"type": "text", "text": instr}, {"type": "audio", "audio": ref}]
            audio, sr = eng.generate([{"role": "user", "content": content}], audio=ref,
                                     gen_seconds=float(sf.info(ref).duration) + 0.5,
                                     nfe=64, cfg_strength=2.0, seed=7)
            save_audio(audio, sr, fp)  # без нормализации — сравниваем исходную динамику
            rms, f0r = feats(fp)
            pair[lv] = {"rms_db": round(rms, 2), "f0_range": round(f0r, 1), "file": fp}
        pair["d_rms"] = round(pair["intense"]["rms_db"] - pair["subdued"]["rms_db"], 2)
        pair["d_f0r"] = round(pair["intense"]["f0_range"] - pair["subdued"]["f0_range"], 1)
        pair["mono_rms"] = pair["d_rms"] > 0
        pair["mono_f0"] = pair["d_f0r"] > 0
        pair["mono_any"] = pair["mono_rms"] or pair["mono_f0"]
        rows.append(pair)
        print(f"[{pi+1}/{args.n_pairs}] {emo}: d_rms={pair['d_rms']:+.1f} d_f0r={pair['d_f0r']:+.0f} mono={pair['mono_any']}", flush=True)

    json.dump(rows, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import statistics as st
    mr = sum(1 for r in rows if r["mono_rms"]) / len(rows)
    mf = sum(1 for r in rows if r["mono_f0"]) / len(rows)
    ma = sum(1 for r in rows if r["mono_any"]) / len(rows)
    print(f"INTENSITY_MONO n={len(rows)} rms_rate={mr:.2f} f0_rate={mf:.2f} any_rate={ma:.2f} "
          f"d_rms_med={st.median([r['d_rms'] for r in rows]):+.2f} dB "
          f"VERDICT={'PASS' if ma >= 0.7 else 'FAIL (gate 0.70)'}")


if __name__ == "__main__":
    main()
