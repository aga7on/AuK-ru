"""Repro for the 'развитие' failure at u11500 ref-mode: seeds x duration x hyphen spelling."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio

    row = json.loads(open(os.path.join(AUK, "local_train", "data", "val.jsonl"), encoding="utf-8").readline())
    stressed = row["messages"][0]["content"][0]["text"].split("'", 2)[1]
    hyphen = stressed.replace("разв+итие", "раз-в+итие")
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []

    def run(name, text, seed, dur):
        instr = f"Say the following with the same voice: '{text}'"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": REF},
        ]}]
        audio, sr = engine.generate(messages, audio=REF, gen_seconds=dur, nfe=64,
                                    cfg_strength=2.0, seed=seed)
        out = os.path.join(args.out, f"rz_{name}_s{seed}_d{dur}.wav")
        save_audio(audio, sr, out)
        manifest.append({"idx": len(manifest), "variant": name, "seed": seed, "dur": dur,
                         "text": stressed, "ref": REF, "gen": out, "gen_seconds": dur})
        print(f"generated {out}", flush=True)

    for seed in (1234, 7, 42):
        for dur in (5.0, 6.0):
            run("default", stressed, seed, dur)
    run("hyphen", hyphen, 1234, 5.0)
    run("hyphen", hyphen, 7, 5.0)

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("RAZV_REPRO_DONE", flush=True)


if __name__ == "__main__":
    main()
