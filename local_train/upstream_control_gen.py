"""Upstream-контроль: генерация 31 задачи оригинального AuK на upstream base и B@500.

Отделяет «модель не умела» (upstream тоже не выполняет) от «адаптация сломала»
(B@500 хуже upstream на той же задаче).

usage:
  python upstream_control_gen.py --variant upstream --ckpt ckpts/AuK/auk_base.safetensors --out local_tests/upstream_control/upstream
  python upstream_control_gen.py --variant s2_B500 --ckpt local_train/run_s2_B/merged/auk_s2_B_500.safetensors --out local_tests/upstream_control/B500
"""
import argparse
import json
import os
import sys
import time

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from auk.infer.infer_auk import AukInfer, save_audio

    manifest_src = os.path.join(AUK, "local_train", "reports", "deepseek_supervised",
                                "original_eval_manifest.json")
    entries = json.load(open(manifest_src, encoding="utf-8"))["entries"]
    if args.limit:
        entries = entries[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"variant={args.variant} entries={len(entries)} ckpt={args.ckpt}", flush=True)

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device=args.device,
        dtype="bf16",
    )

    results = []
    t0 = time.time()
    for i, e in enumerate(entries):
        fid = f"e{i:02d}_{e['task'].lower().replace(' ', '_').replace('(', '').replace(')', '')[:40]}"
        out = os.path.join(args.out, fid + ".wav")
        audio_ref = e.get("audio")
        content = [{"type": "text", "text": e["instruction"]}]
        if audio_ref:
            content.append({"type": "audio", "audio": audio_ref})
        messages = [{"role": "user", "content": content}]
        try:
            audio, sr = engine.generate(messages, audio=(audio_ref or None),
                                        gen_seconds=float(e["gen_seconds"]),
                                        nfe=64, cfg_strength=2.0, seed=args.seed)
            save_audio(audio, sr, out)
            results.append({"id": fid, "group": e["group"], "task": e["task"],
                            "instruction": e["instruction"], "ref": audio_ref,
                            "file": out, "gen_seconds": e["gen_seconds"], "status": "ok"})
            print(f"[{i+1}/{len(entries)}] {fid} ok ({(time.time()-t0)/60:.1f}m)", flush=True)
        except Exception as ex:
            results.append({"id": fid, "group": e["group"], "task": e["task"],
                            "instruction": e["instruction"], "ref": audio_ref,
                            "file": None, "status": f"error {type(ex).__name__}: {ex}"})
            print(f"[{i+1}/{len(entries)}] {fid} FAILED {type(ex).__name__}", flush=True)

    rp = os.path.join(args.out, "results.json")
    json.dump({"variant": args.variant, "ckpt": args.ckpt, "seed": args.seed,
               "n": len(results), "ok": sum(1 for r in results if r["status"] == "ok"),
               "results": results}, open(rp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"UPSTREAM_CONTROL_DONE variant={args.variant} ok="
          f"{sum(1 for r in results if r['status'] == 'ok')}/{len(results)} -> {rp}", flush=True)


if __name__ == "__main__":
    main()
