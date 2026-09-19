"""Stress-marking variants for the 'Африканском Роге' word, to find the formulation the model obeys."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

BASE = "Ещё более важную роль на Африканском {W} играет устойчивое развитие."
PLAIN = "Ещё более важную роль на Африканском Роге играет устойчивое развитие."
REF = os.path.join(AUK, "local_tests", "user_ref.wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer.ru_translit import ru_to_latin

    variants = {
        "v0_Rplus_oge": BASE.replace("{W}", "Р+оге"),
        "v1_Ro_plus_ge": BASE.replace("{W}", "Ро+ге"),
        "v2_lower_r_plus_oge": BASE.replace("{W}", "р+оге"),
        "v3_plain_Roge": BASE.replace("{W}", "Роге"),
        "v4_translit": ru_to_latin(BASE.replace("{W}", "Р+оге")),
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
        instr = f"Say the following with the same voice: '{text}'"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": REF},
        ]}]
        audio, sr = engine.generate(messages, audio=REF, gen_seconds=5.5, nfe=32,
                                    cfg_strength=2.0, seed=args.seed)
        out = os.path.join(args.out, f"{name}.wav")
        save_audio(audio, sr, out)
        manifest.append({"idx": len(manifest), "variant": name, "text": PLAIN, "ref": REF,
                         "gen": out, "gen_seconds": 5.5, "instruction_text": text})
        print(f"generated {out}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("STRESS_VARIANTS_DONE", flush=True)


if __name__ == "__main__":
    main()
