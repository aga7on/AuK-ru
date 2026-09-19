"""Generate evaluation samples with a given AuK checkpoint (base or LoRA-merged) on one GPU."""
import argparse
import json
import os
import sys

AUK_ROOT = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK_ROOT, "src"))

PROMPTS = [
    "Привет! Это проверка русского произношения. Раз, два, три.",
    "Сегодня отличный день, и мы тестируем новую систему синтеза.",
    "Замок стоит на горе, а на двери висит большой замок.",
    "Съешь ещё этих мягких французских булок, да выпей чаю.",
    "Мы проверяем ударения: договор, каталог, звонит, красивее.",
    "Он сказал: «Я вернусь завтра утром», и вышел из комнаты.",
    "Вся семья была дома, и день прошёл отлично.",
    "Объявление: объём работы большой, но пять друзей съели весь борщ.",
]
REFS = ([r"G:\AI\AuK\local_tests\user_ref.wav"] * 4
        + [r"G:\AI\AuK\local_tests\ref_ru.wav"] * 4)


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


GEN_SECONDS = [estimate_seconds(p) for p in PROMPTS]


def accentize_and_translit(text):
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer.ru_translit import ru_to_latin

    return ru_to_latin(_accentize_ru(text))


def build_instruction_legacy(text):
    return f"Say the following with the same voice: '{accentize_and_translit(text)}'"


def build_instruction(text, representation):
    if representation == "cyr_stress":
        from auk.infer.infer_gradio import _accentize_ru

        body = _accentize_ru(text)
    elif representation == "cyr_plain":
        body = text
    else:  # translit
        from auk.infer.infer_gradio import _accentize_ru
        from auk.infer.ru_translit import ru_to_latin

        body = ru_to_latin(_accentize_ru(text))
    return f"Say the following with the same voice: '{body}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--representation", default="cyr_stress", choices=["translit", "cyr_stress", "cyr_plain"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK_ROOT, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device=args.device,
        dtype="bf16",
    )
    manifest = []
    for i, (text, ref, secs) in enumerate(zip(PROMPTS, REFS, GEN_SECONDS)):
        instr = build_instruction(text, args.representation)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=args.seed)
        out = os.path.join(args.out_dir, f"gen_{i}.wav")
        save_audio(audio, sr, out)
        manifest.append({"idx": i, "text": text, "ref": ref, "gen": out, "gen_seconds": secs,
                         "representation": args.representation, "instruction": instr})
        print(f"generated {out}", flush=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("EVAL_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
