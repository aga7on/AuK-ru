"""Best-of-N step 1: generate N candidates per phrase (different seeds) through product path."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = r"G:\AI\AuK\local_tests\user_ref.wav"

PHRASES = [
    "Ещё более важную роль на Африканском Роге играет устойчивое развитие.",
    "Пожалуйста, не опаздывайте на встречу.",
    "Две тысячи двадцать шестого года мы запустили этот проект.",
    "Мы проверяем ударения: договор, каталог, звонит, красивее.",
    "Сельдь под шубой — традиционное новогоднее блюдо.",
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1234,7,42")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.join(args.out, "cand"), exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    for idx, text in enumerate(PHRASES):
        body = _accentize_ru(text)
        instr = f"Say the following with the same voice: '{body}'"
        secs = estimate_seconds(text)
        for seed in seeds:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, "cand", f"c{idx:02d}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": idx, "text": text, "seed": seed, "gen": out,
                             "ref": REF, "gen_seconds": secs, "instruction": instr})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("BESTOFN_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
