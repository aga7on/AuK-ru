"""Soft-sign convention A/B v2 for «роль»: latin caron, mixed cyrillic ь, cyrillic word, digraphs."""
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
    from auk.infer.ru_translit import ru_to_latin

    row = json.loads(open(os.path.join(AUK, "local_train", "data", "val.jsonl"), encoding="utf-8").readline())
    stressed = row["messages"][0]["content"][0]["text"].split("'", 2)[1]
    base = ru_to_latin(stressed)
    if "rol'" not in base:
        print("WARN: 'rol'' not found in base translit")
    variants = {
        "apos": base,
        "y_digraph": base.replace("rol'", "roly"),
        "caron": base.replace("rol'", "ro\u013e"),      # roľ (l with caron)
        "mixed_cyrь": base.replace("rol'", "rol\u044c"),  # latin + Cyrillic ь
        "cyr_word": base.replace("rol'", "\u0440\u043e\u043b\u044c"),  # роль (Cyrillic word)
        "apos_j": base.replace("rol'", "rol'j"),
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
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=5.5, nfe=64,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, f"soft2_{name}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": len(manifest), "variant": name, "seed": seed, "text": stressed,
                             "instruction_text": text, "ref": REF, "gen": out, "gen_seconds": 5.5})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("SOFTL2_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
