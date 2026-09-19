"""Instruct-mode representation A/B: cyrillic+stress vs cyrillic plain vs translit (no reference audio)."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer.ru_translit import ru_to_latin

    row = json.loads(open(os.path.join(AUK, "local_train", "data", "val.jsonl"), encoding="utf-8").readline())
    stressed = row["messages"][0]["content"][0]["text"].split("'", 2)[1]
    plain = stressed.replace("+", "").replace("'", "")
    variants = {
        "instr_cyr_stress": stressed,
        "instr_cyr_plain": plain,
        "instr_translit": ru_to_latin(stressed),
    }
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    judge_ref = os.path.join(AUK, "local_tests", "user_ref.wav")
    manifest = []
    for name, text in variants.items():
        for seed in (1234, 7):
            instr = f"Say the following in Russian with clear, natural pronunciation: '{text}'"
            messages = [{"role": "user", "content": [{"type": "text", "text": instr}]}]
            audio, sr = engine.generate(messages, audio=None, gen_seconds=5.5, nfe=64,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, f"{name}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": len(manifest), "variant": name, "seed": seed, "text": stressed,
                             "instruction_text": instr, "ref": judge_ref, "gen": out, "gen_seconds": 5.5})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("INSTRUCT_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
