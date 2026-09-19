"""Запуск оценочного пака на заданном чекпоинте (u0-контроль и A/B).

Протокол A/B: trim=False, bestofn=1, NFE=64, CFG=2.0, seed=1234.
usage: run_eval_pack.py --pack <pack.json> --ckpt <merged.safetensors> --out <dir>
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
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from auk.infer.infer_auk import AukInfer, save_audio

    pack = json.load(open(args.pack, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    print(f"pack: {len(pack)} entries | ckpt: {os.path.basename(args.ckpt)}", flush=True)

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device=args.device,
        dtype="bf16",
    )

    manifest = []
    t0 = time.time()
    for i, item in enumerate(pack):
        ref = item.get("ref")
        content = [{"type": "text", "text": item["instruction"]}]
        if ref:
            content.append({"type": "audio", "audio": ref})
        messages = [{"role": "user", "content": content}]
        try:
            audio, sr = engine.generate(messages, audio=(ref or None),
                                        gen_seconds=float(item["gen_seconds"]),
                                        nfe=64, cfg_strength=2.0, seed=args.seed)
            # БЕЗ trim (протокол)
            out = os.path.join(args.out, f"{item['id']}.wav")
            save_audio(audio, sr, out)
            row = {"id": item["id"], "kind": item["kind"], "instruction": item["instruction"],
                   "ref": ref, "file": out, "gen_seconds": item["gen_seconds"],
                   "text": item.get("text")}
            manifest.append(row)
            print(f"[{i+1}/{len(pack)}] {item['id']} ok ({(time.time()-t0)/60:.1f}m)", flush=True)
        except Exception as e:
            manifest.append({"id": item["id"], "kind": item["kind"], "error": f"{type(e).__name__}: {e}"})
            print(f"[{i+1}/{len(pack)}] {item['id']} FAILED {type(e).__name__}", flush=True)

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"EVAL_PACK_DONE: {len(manifest)} entries in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
