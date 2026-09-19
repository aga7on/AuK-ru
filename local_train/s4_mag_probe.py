"""Magnitude-пробник s4: генерация tool-инструкций с РАЗНЫМИ величинами + DSP-замер.

Проверяет, научилась ли модель отображать "величина в инструкции -> величина эффекта"
(диагноз s3: volume_up +12.12 вместо +6, pitch +3.91 вместо +2).
Референсы — из u0_control manifest (speech-корпус, НЕ входили в tools-пул обучения).

usage:
  python s4_mag_probe.py --ckpt <merged.safetensors> --out local_tests/s4_mag_probe
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_train")
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, LT)

INSTR = {
    ("volume_up", 3): "Raise the volume by 3 decibels",
    ("volume_up", 6): "Raise the volume by 6 decibels",
    ("volume_up", 9): "Raise the volume by 9 decibels",
    ("volume_down", -3): "Lower the volume by 3 decibels",
    ("volume_down", -6): "Lower the volume by 6 decibels",
    ("volume_down", -9): "Lower the volume by 9 decibels",
    ("pitch_up", 1): "Raise the pitch by 1 semitone",
    ("pitch_up", 2): "Raise the pitch by 2 semitones",
    ("pitch_up", 3): "Raise the pitch by 3 semitones",
    ("pitch_down", -1): "Lower the pitch by 1 semitone",
    ("pitch_down", -2): "Lower the pitch by 2 semitones",
    ("pitch_down", -3): "Lower the pitch by 3 semitones",
    ("speed_up", 1.1): "Increase the speech speed by 1.1 times",
    ("speed_up", 1.2): "Increase the speech speed by 1.2 times",
    ("speed_down", 0.9): "Lower the speech speed to 0.9 times",
    ("speed_down", 0.8): "Lower the speech speed to 0.8 times",
}
N_PER = 4  # 16 комбинаций x 4 = 64 генерации


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    man = json.load(open(os.path.join(AUK, "local_tests", "u0_control", "manifest.json"),
                         encoding="utf-8"))
    refs = [m["ref"] for m in man if m.get("kind") == "tool" and m.get("ref")]
    rng = random.Random(args.seed)
    rng.shuffle(refs)

    os.makedirs(args.out, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device=args.device,
        dtype="bf16",
    )

    # скорость задаём через gen_seconds: секунды источника / rate
    from tool_dsp_measure import active_stats, load

    tasks = []
    ri = 0
    for (op, mag), instr in INSTR.items():
        for k in range(N_PER):
            ref = refs[ri % len(refs)]
            ri += 1
            tid = f"{op}_{str(mag).replace('-', 'm').replace('.', 'p')}_{k}"
            tasks.append({"id": tid, "op": op, "mag": mag, "instr": instr, "ref": ref})

    rows = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        out = os.path.join(args.out, t["id"] + ".wav")
        try:
            xin, sr = load(t["ref"])
            si = active_stats(xin, sr)
            dur = si["dur_s"] if si["dur_s"] else len(xin) / sr
            rate = t["mag"] if t["op"].startswith("speed") else 1.0
            audio, gsr = engine.generate(
                [{"role": "user", "content": [
                    {"type": "text", "text": t["instr"]},
                    {"type": "audio", "audio": t["ref"]}]}],
                audio=t["ref"], gen_seconds=float(dur / rate) + 0.5,
                nfe=64, cfg_strength=2.0, seed=args.seed)
            save_audio(audio, gsr, out)
            rows.append({"id": t["id"], "op": t["op"], "mag": t["mag"], "ref": t["ref"],
                         "file": out, "status": "ok"})
        except Exception as ex:
            rows.append({"id": t["id"], "op": t["op"], "mag": t["mag"], "ref": t["ref"],
                         "file": None, "status": f"error {type(ex).__name__}: {ex}"})
        if (i + 1) % 8 == 0:
            print(f"[{i+1}/{len(tasks)}] {(time.time()-t0)/60:.1f}m", flush=True)

    # DSP-замер
    from tool_dsp_measure import median_f0
    for r in rows:
        if r["status"] != "ok":
            continue
        try:
            xin, sr = load(r["ref"])
            xout, _ = load(r["file"])
        except Exception:
            r["valid"] = False
            r["reason"] = "load"
            continue
        si, so = active_stats(xin, sr), active_stats(xout, sr)
        if r["op"].startswith("volume"):
            if si["rms_db"] is not None and so["rms_db"] is not None:
                r["delta_db"] = round(so["rms_db"] - si["rms_db"], 2)
                r["err_db"] = round(r["delta_db"] - r["mag"], 2)
        elif r["op"].startswith("speed"):
            if si["dur_s"] and so["dur_s"]:
                r["rate"] = round(si["dur_s"] / so["dur_s"], 3)
                r["err_rate"] = round(r["rate"] - r["mag"], 3)
        else:
            fi, ni = median_f0(xin, sr)
            fo, no = median_f0(xout, sr)
            if fi and fo and ni >= 30 and no >= 30:
                while fo > fi * 2:
                    fo /= 2
                while fo < fi / 2:
                    fo *= 2
                r["semitones"] = round(12 * np.log2(fo / fi), 2)
                r["err_st"] = round(r["semitones"] - r["mag"], 2)

    # агрегат
    agg = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = f"{r['op']}@{r['mag']}"
        m = "delta_db" if r["op"].startswith("volume") else (
            "semitones" if r["op"].startswith("pitch") else "rate")
        if m in r:
            agg.setdefault(key, []).append(r[m])
    summary = {k: {"n": len(v), "mean": round(float(np.mean(v)), 3),
                   "err_mean": round(float(np.mean(v)) - float(k.split("@")[1]), 3)}
               for k, v in agg.items()}

    json.dump({"ckpt": args.ckpt, "seed": args.seed, "n": len(rows),
               "ok": sum(1 for r in rows if r["status"] == "ok"),
               "summary": summary, "rows": rows},
              open(os.path.join(args.out, "mag_probe.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"MAG_PROBE_DONE ok={sum(1 for r in rows if r['status'] == 'ok')}/{len(rows)}")


if __name__ == "__main__":
    main()
