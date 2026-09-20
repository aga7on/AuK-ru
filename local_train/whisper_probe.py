# -*- coding: utf-8 -*-
"""Whisper-проба: проверяет, следует ли модель speaking_mode-инструкции.

Генерирует whisper-инструкции (2 формата: v7-стиль «with a whispering tone» и
v12-стиль «say in Russian, whispering») + normal-контроль, меряет RMS dB активного
сегмента: шёпот должен быть ТИШЕ normal (whisper_audit: teacher-шёпоты −7.5 дБ).
Объективный критерий «шёпот похож на шёпот»: RMS_whisper < RMS_normal − 2 дБ.

usage: python whisper_probe.py --ckpt <merged> --config <cfg> --out <dir> [--device cuda:1]
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
TEXTS = [
    "Петрозаводск, проспект Ленина, дом пятнадцать.",
    "Я скажу тебе по секрету, никому не рассказывай.",
    "Библиотека закрывается через десять минут.",
    "Тихо, ребёнок только что уснул в соседней комнате.",
    "Шёпотом передай соседу, что пора сдавать работу.",
]


def rms_db(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    act = x[np.abs(x) > 0.005]
    if len(act) == 0:
        return -60.0
    return float(20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    ref = next(e["ref"] for e in pack if e.get("kind") == "clone" and e["id"] == "clone01")

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    variants = {
        "normal": "Reproduce the reference voice and say in Russian: '{t}'",
        "whisper_tone": "Reproduce the reference voice and say in Russian with a whispering tone: '{t}'",
        "whisper_mode": "Reproduce the reference voice and say in Russian, whispering: '{t}'",
    }
    results = []
    for vi, (vname, tmpl) in enumerate(variants.items()):
        for ti, t in enumerate(TEXTS):
            instr = tmpl.format(t=t)
            fp = os.path.join(args.out, f"{vname}_t{ti}.wav")
            content = [{"type": "text", "text": instr}, {"type": "audio", "audio": ref}]
            audio, sr = eng.generate([{"role": "user", "content": content}], audio=ref,
                                     gen_seconds=float(sf.info(ref).duration) + 0.5,
                                     nfe=64, cfg_strength=2.0, seed=7)
            save_audio(audio, sr, fp)  # БЕЗ normalize — нам нужна исходная громкость
            results.append({"variant": vname, "text": t, "file": fp, "rms_db": round(rms_db(fp), 2)})
            print(results[-1], flush=True)

    json.dump(results, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    import statistics as st
    by = {}
    for r in results:
        by.setdefault(r["variant"], []).append(r["rms_db"])
    print("--- RMS dB by variant ---")
    for v, xs in by.items():
        print(f"{v}: median {st.median(xs):.1f} dB")
    norm = st.median(by.get("normal", [0]))
    for v in ("whisper_tone", "whisper_mode"):
        if v in by:
            d = st.median(by[v]) - norm
            print(f"{v} vs normal: {d:+.1f} dB -> {'WHISPER-LIKE' if d <= -2 else 'NOT quieter (scream/normal)'}")


if __name__ == "__main__":
    main()
