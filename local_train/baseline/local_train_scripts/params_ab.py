"""Sampling-parameter A/B for micro-pause issue: nfe / cfg / sway variations, seeds 1234 and 7."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

PHRASES = [
    ("Иван Петров работает в банке с две тысячи двадцать шестого года.", 7.5),
    ("Позвони мне, пожалуйста, когда освободишься.", 6.0),
]
REF = os.path.join(AUK, "local_tests", "user_ref.wav")

SETTINGS = [
    ("cur_nfe32_cfg2_sway-1", 32, 2.0, -1.0, 1.0),
    ("nfe64", 64, 2.0, -1.0, 1.0),
    ("sway0", 32, 2.0, 0.0, 1.0),
    ("cfg3", 32, 3.0, -1.0, 1.0),
    ("dur085", 32, 2.0, -1.0, 0.85),
    ("dur115", 32, 2.0, -1.0, 1.15),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

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
    for pi, (phrase, secs) in enumerate(PHRASES):
        body = _accentize_ru(phrase)
        for name, nfe, cfg, sway, durf in SETTINGS:
            eff_secs = round(secs * durf, 1)
            for seed in [1234, 7]:
                instr = f"Say the following with the same voice: '{body}'"
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": instr},
                    {"type": "audio", "audio": REF},
                ]}]
                audio, sr = engine.generate(messages, audio=REF, gen_seconds=eff_secs, nfe=nfe,
                                            cfg_strength=cfg, sway_sampling_coef=sway, seed=seed)
                out = os.path.join(args.out, f"par_p{pi}_{name}_s{seed}.wav")
                save_audio(audio, sr, out)
                manifest.append({"idx": len(manifest), "setting": name, "seed": seed, "phrase": pi,
                                 "text": phrase, "ref": REF, "gen": out, "gen_seconds": eff_secs})
                print(f"generated {out}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("PARAMS_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
