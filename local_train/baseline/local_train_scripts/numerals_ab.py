"""Numeral A/B: words-with-stress vs digits vs words-plain for the date phrase."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

BASE_WORDS = "Иван Петров работает в банке с две тысячи двадцать шестого года."
REF = os.path.join(AUK, "local_tests", "user_ref.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    stressed = _accentize_ru(BASE_WORDS)
    plain = stressed.replace("+", "").replace("'", "")
    digits = "Иван Петров работает в банке с 2026 года."
    variants = {
        "words_stress": stressed,
        "words_plain": plain,
        "digits": digits,
        "digits_god": "Иван Петров работает в банке с 2026-го года.",
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
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=7.5, nfe=32,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, f"num_{name}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": len(manifest), "variant": name, "seed": seed, "text": BASE_WORDS,
                             "instruction_text": text, "ref": REF, "gen": out, "gen_seconds": 7.5})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("NUMERALS_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
