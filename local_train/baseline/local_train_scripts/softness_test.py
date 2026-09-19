"""Softness A/B: words with ь/ъ in cyrillic-stress vs translit representation (best stage-0 ckpt)."""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

CKPT = os.path.join(AUK, "local_train", "run_ru", "merged", "auk_ru_3000.safetensors")
REF = os.path.join(AUK, "local_tests", "user_ref.wav")
OUT = os.path.join(AUK, "local_tests", "softness")

PROMPTS = [
    "День был тёплый, и вся семья пошла гулять.",
    "Он объяснил, что объём работы очень большой.",
    "Пять друзей съели весь борщ и выпили чай.",
]


def build_instruction(text, representation):
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer.ru_translit import ru_to_latin

    accented = _accentize_ru(text)
    body = accented if representation == "cyr_stress" else ru_to_latin(accented)
    return f"Say the following with the same voice: '{body}'"


def main():
    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(CKPT), "config.yaml"),
        ckpt_path=CKPT,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    for rep in ["cyr_stress", "translit"]:
        for i, text in enumerate(PROMPTS):
            instr = build_instruction(text, rep)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=5.5, nfe=32,
                                        cfg_strength=2.0, seed=42)
            out = os.path.join(OUT, f"softness_{rep}_{i}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": i, "text": text, "ref": REF, "gen": out, "gen_seconds": 5.5,
                             "representation": rep, "instruction": instr})
            print(f"generated {out}", flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("SOFTNESS_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
