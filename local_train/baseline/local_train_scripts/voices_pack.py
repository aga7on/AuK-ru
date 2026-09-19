"""Voices pack: 4 reference voices x 4 phrases, seed 1234, cyrillic+stress."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

PHRASES = [
    ("Позвони мне, пожалуйста, когда освободишься.", 6.0),
    ("Съешь ещё этих мягких французских булок, да выпей чаю.", 7.0),
    ("Достопримечательности Санкт-Петербурга впечатляют туристов.", 7.5),
    ("Электричка отправляется через пятнадцать минут с третьего пути.", 7.5),
]
REFS = [
    ("own_m", os.path.join(AUK, "local_tests", "user_ref.wav")),
    ("male2", os.path.join(AUK, "local_tests", "ref_male2.wav")),
    ("female1", os.path.join(AUK, "local_tests", "ref_ru.wav")),
    ("female2", os.path.join(AUK, "local_tests", "ref_female2.wav")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
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
    for vname, ref in REFS:
        for i, (text, secs) in enumerate(PHRASES):
            body = _accentize_ru(text)
            instr = f"Say the following with the same voice: '{body}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": ref},
            ]}]
            audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=32,
                                        cfg_strength=2.0, seed=args.seed)
            out = os.path.join(args.out, f"vol_{vname}_p{i}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": i, "voice": vname, "text": text, "ref": ref, "gen": out,
                             "gen_seconds": secs, "representation": "cyr_stress", "seed": args.seed})
            print(f"generated {out}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("VOICES_PACK_DONE", flush=True)


if __name__ == "__main__":
    main()
