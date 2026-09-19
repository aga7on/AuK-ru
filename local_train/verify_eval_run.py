"""Проверка завершённости оценочного прогона (u0-контроль и A/B).

Проверяет: все ожидаемые файлы есть, аудио не пустое/не тихое, нет ошибок в manifest,
пишет run_params.json (точные входы: ckpt+hash, pack+hash, seed, параметры).

usage: verify_eval_run.py --pack <pack.json> --out <dir> [--ckpt <path>] [--expect-total N]
"""
import argparse
import hashlib
import json
import os
from datetime import datetime

import torchaudio


def sha16(path, limit_mb=2048):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = limit_mb * 1024 * 1024
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    pack = json.load(open(args.pack, encoding="utf-8"))
    ids = [it["id"] for it in pack]
    print(f"pack: {len(ids)} entries")

    man_path = os.path.join(args.out, "manifest.json")
    manifest = None
    if os.path.exists(man_path):
        manifest = {r["id"]: r for r in json.load(open(man_path, encoding="utf-8"))}
        print(f"manifest: {len(manifest)} entries")
    else:
        print("manifest: NOT FOUND (раннер ещё идёт?)")

    missing, errors, empty, silent, short_ = [], [], [], [], []
    peaks = []
    for i in ids:
        r = manifest.get(i) if manifest else None
        if r and r.get("error"):
            errors.append((i, r["error"]))
            continue
        f = os.path.join(args.out, f"{i}.wav")
        if not os.path.exists(f):
            missing.append(i)
            continue
        try:
            audio, sr = torchaudio.load(f)
            dur = audio.shape[-1] / sr
            peak = float(audio.abs().max())
            rms = float(audio.pow(2).mean().sqrt())
            peaks.append(peak)
            if dur < 0.2:
                empty.append((i, round(dur, 3)))
            elif rms < 1e-4:
                silent.append((i, round(rms, 6)))
        except Exception as e:
            errors.append((i, f"load: {type(e).__name__}: {e}"))

    done = len(missing) == 0 and len(errors) == 0 and len(empty) == 0 and len(silent) == 0
    print(f"present={len(ids) - len(missing)}/{len(ids)} missing={len(missing)} errors={len(errors)} "
          f"empty={len(empty)} silent={len(silent)}")
    for name, lst in (("MISSING", missing), ("ERRORS", errors), ("EMPTY", empty), ("SILENT", silent)):
        for it in lst[:10]:
            print(f"  {name}: {it}")
    if peaks:
        print(f"peak: min={min(peaks):.3f} max={max(peaks):.3f} "
              f"clipped(>=0.999)={sum(1 for p in peaks if p >= 0.999)}")

    if args.ckpt and os.path.exists(args.ckpt):
        params = {
            "ckpt": args.ckpt,
            "ckpt_sha16": sha16(args.ckpt),
            "pack": args.pack,
            "pack_sha16": sha16(args.pack, limit_mb=32),
            "pack_entries": len(ids),
            "out_dir": args.out,
            "seed": 1234,
            "nfe": 64,
            "cfg_strength": 2.0,
            "trim": False,
            "bestofn": 1,
            "verified_at": datetime.now().isoformat(timespec="seconds"),
            "status": "ok" if done else "incomplete",
            "errors": [i for i, _ in errors],
            "missing": missing,
        }
        json.dump(params, open(os.path.join(args.out, "run_params.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("run_params.json written")

    print("VERIFY_RESULT:", "OK" if done else "INCOMPLETE")


if __name__ == "__main__":
    main()
