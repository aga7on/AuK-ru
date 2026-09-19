"""Independent runner for the authorized 9-WAV diagnostic (3 fixtures x 3 controls).

Properties required by root:
- uses only the public AukInfer API, explicit config/ckpt paths, .venv;
- identical seed=1234, nfe=64, cfg=2.0, trim=false, bestofn=1;
- no clipping normalization, no silence trimming; raw float output saved;
- one engine at a time, model released between controls;
- atomic per-item progress manifest, errors/no_result preserved;
- explicit model/source/pack/code hashes and run params;
- does not import or reuse the existing run_eval_pack.py (avoids its null-float path).

CLI: python run_diag.py [--device cuda:1] [--only diag_pitch_en_up2,diag_zeroshot_en]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
PACK_PATH = os.path.join(HERE, "diag_pack.json")
MODELS_SHA = os.path.join(HERE, "diag_models_sha.json")
QWEN = os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
        rows = []
        for line in out:
            idx, free, used, util = [p.strip() for p in line.split(",")]
            rows.append({"index": int(idx), "free_mib": int(free), "used_mib": int(used),
                         "util_pct": int(util)})
        return rows
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def choose_device(explicit: str | None, snap: list[dict]) -> str:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if explicit:
        return explicit
    valid = [r for r in snap if "free_mib" in r]
    if not valid:
        return "cuda:0"
    best = max(valid, key=lambda r: (r["free_mib"], -r["util_pct"]))
    return f"cuda:{best['index']}"


def model_hashes(pack: dict, device_note: str) -> dict:
    if os.path.isfile(MODELS_SHA):
        cached = json.load(open(MODELS_SHA, encoding="utf-8"))
        if all(cached.get(name, {}).get("sha256_full") == c["sha256_full_expected"]
               for name, c in pack["controls"].items()):
            return cached
    result = {}
    for name, c in pack["controls"].items():
        t0 = time.time()
        got = sha256(c["ckpt"])
        result[name] = {
            "ckpt": c["ckpt"], "sha256_full": got,
            "sha256_full_expected": c["sha256_full_expected"],
            "matches_manifest": got == c["sha256_full_expected"],
            "size": os.path.getsize(c["ckpt"]), "hash_seconds": round(time.time() - t0, 1),
        }
    json.dump(result, open(MODELS_SHA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return result


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def build_messages(fx: dict):
    content = [{"type": "text", "text": fx["instruction"]}]
    if fx.get("ref"):
        content.append({"type": "audio", "audio": fx["ref"]})
    return [{"role": "user", "content": content}]


def probe_output(path: str) -> dict:
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return {
        "output_sha256": sha256(path),
        "sr": int(sr),
        "frames": int(len(x)),
        "duration_s": round(len(x) / sr, 4),
        "peak": round(float(np.max(np.abs(x))), 5) if x.size else None,
        "clip_ratio_ge_0.985": round(float(np.mean(np.abs(x) >= 0.985)), 6) if x.size else None,
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 2) if x.size else None,
        "nonzero": bool(np.any(x != 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="comma list of fixture ids")
    ap.add_argument("--controls", default=None, help="comma list of control names")
    args = ap.parse_args()

    pack = json.load(open(PACK_PATH, encoding="utf-8"))
    if pack.get("validation_errors"):
        raise SystemExit(f"pack has validation errors: {pack['validation_errors']}")

    only = set(args.only.split(",")) if args.only else None
    ctrl_filter = set(args.controls.split(",")) if args.controls else None

    snap = gpu_snapshot()
    device = choose_device(args.device, snap)
    print(f"GPU snapshot: {snap}")
    print(f"chosen device: {device}")

    model_sha = model_hashes(pack, device)
    for name, m in model_sha.items():
        print(f"model {name}: sha256 matches manifest = {m['matches_manifest']} "
              f"({m['hash_seconds']}s to hash)")

    out_root = pack["output_root"]
    manifest_path = os.path.join(out_root, "progress_manifest.json")
    os.makedirs(out_root, exist_ok=True)
    manifest = {
        "schema": "diag-run-1",
        "pack": PACK_PATH,
        "pack_sha256": sha256(PACK_PATH),
        "run_config": pack["run_config"],
        "device": device,
        "gpu_snapshot_start": snap,
        "model_sha256": model_sha,
        "code_sha256": {k: sha256(os.path.join(HERE, k)) for k in ("build_diag_pack.py", "run_diag.py")},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": [],
        "summary": {},
    }
    atomic_write_json(manifest_path, manifest)

    import torch
    from auk.infer.infer_auk import AukInfer, save_audio

    run_t0 = time.time()
    for cname, c in pack["controls"].items():
        if ctrl_filter and cname not in ctrl_filter:
            continue
        engine = None
        load_t0 = time.time()
        print(f"\n=== control {cname} : loading engine on {device} ===")
        try:
            engine = AukInfer(config_path=c["config"], ckpt_path=c["ckpt"],
                              qwen_path=QWEN, cpu_offload=True, device=device, dtype="bf16")
        except Exception as exc:  # noqa: BLE001
            for fx in pack["fixtures"]:
                if only and fx["id"] not in only:
                    continue
                manifest["items"].append({
                    "control": cname, "fixture": fx["id"], "status": "no_result",
                    "error": f"engine_load: {type(exc).__name__}: {exc}"})
            atomic_write_json(manifest_path, manifest)
            print(f"ENGINE LOAD FAILED for {cname}: {exc}")
            continue
        load_s = round(time.time() - load_t0, 1)
        print(f"engine loaded in {load_s}s")

        for fx in pack["fixtures"]:
            if only and fx["id"] not in only:
                continue
            item = {"control": cname, "fixture": fx["id"], "task": fx["task"],
                    "language": fx["language"], "instruction": fx["instruction"],
                    "ref": fx.get("ref"), "gen_seconds": fx["gen_seconds"],
                    "engine_load_s": load_s, "device": device}
            t0 = time.time()
            try:
                messages = build_messages(fx)
                audio, sr = engine.generate(
                    messages,
                    audio=(fx.get("ref") or None),
                    gen_seconds=float(fx["gen_seconds"]),
                    nfe=64,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                    seed=1234,
                )
                out_dir = os.path.join(out_root, cname)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{fx['id']}.wav")
                save_audio(audio, sr, out_path)  # raw, no trim/normalize
                item.update({"status": "ok", "output": out_path,
                             "gen_seconds_actual": round(audio.shape[-1] / sr, 4)})
                item.update(probe_output(out_path))
            except Exception as exc:  # noqa: BLE001
                item.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {fx['id']}: ERROR {exc}")
            item["runtime_s"] = round(time.time() - t0, 1)
            manifest["items"].append(item)
            atomic_write_json(manifest_path, manifest)  # atomic per-item progress
            if item["status"] == "ok":
                print(f"  {fx['id']}: ok dur={item['duration_s']}s "
                      f"peak={item['peak']} clip={item['clip_ratio_ge_0.985']} "
                      f"({item['runtime_s']}s)")

        # release engine between controls
        try:
            del engine
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        print(f"released engine {cname}")

    ok = [i for i in manifest["items"] if i["status"] == "ok"]
    outputs = [i["output"] for i in ok]
    manifest["summary"] = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_items": len(manifest["items"]),
        "ok": len(ok),
        "error": sum(1 for i in manifest["items"] if i["status"] == "error"),
        "no_result": sum(1 for i in manifest["items"] if i["status"] == "no_result"),
        "unique_outputs": len(set(outputs)),
        "unique_output_sha256": len(set(i["output_sha256"] for i in ok)),
        "wall_seconds": round(time.time() - run_t0, 1),
    }
    atomic_write_json(manifest_path, manifest)
    print("\nSUMMARY", json.dumps(manifest["summary"], ensure_ascii=False))
    if manifest["summary"]["unique_outputs"] != len(outputs):
        print("WARNING: duplicate output paths detected")


if __name__ == "__main__":
    main()
