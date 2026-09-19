"""Name-mangling A/B: how to spell «Петров» so the model keeps the т (per-seed runs)."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")
TAIL = "работает в банке с две тысячи двадцать шестого года."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    default = _accentize_ru("Иван Петров " + TAIL)
    variants = {
        "default_stress": default,
        "plain": default.replace("Петр+ов", "Петров"),
        "caps_stress": default.replace("Петр+ов", "ПетрОв"),
        "latin": default.replace("Петр+ов", "Petrov"),
    }
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    for name, text in variants.items():
        for seed in (1234, 7):
            instr = f"Say the following with the same voice: '{text}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=7.5, nfe=64,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, f"name_{name}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": len(manifest), "variant": name, "seed": seed,
                             "text": "Иван Петров " + TAIL, "instruction_text": text,
                             "ref": REF, "gen": out, "gen_seconds": 7.5})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("NAME_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
