"""Round-2 respell tests for stubborn words (произношения, смотреть, весенний, решений)."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = r"G:\AI\AuK\local_tests\user_ref.wav"

CASES = [
    ("Привет! Это проверка русского произношения. Раз, два, три.",
     [("v0_default", None, None),
      ("v1_pro-izno-shenia", "произнош+ения", "про-изно-ш+ения"),
      ("v2_proizno-shenia", "произнош+ения", "произно-ш+ения"),
      ("v3_pra-iz-nashenia", "произнош+ения", "пра-из-наш+ения")]),
    ("Мне нравится смотреть на звёзды летними ночами.",
     [("v0_default", None, None),
      ("v1_smo-tret", "смотр+еть", "смо-тр+еть"),
      ("v2_smatret_phonetic", "смотр+еть", "сматр+еть")]),
    ("Люблю грозу в начале мая, когда весенний первый гром.",
     [("v0_default", None, None),
      ("v1_vessenii", "вес+енний", "весс+енний")]),
    ("Сложные времена требуют простых решений.",
     [("v0_default", None, None),
      ("v1_reshennii", "реш+ений", "реш+енний")]),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


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
    for pi, (text, variants) in enumerate(CASES):
        stressed = _accentize_ru(text)
        secs = estimate_seconds(text)
        for name, find, repl in variants:
            body = stressed if find is None else stressed.replace(find, repl)
            assert find is None or body != stressed, f"replace failed: {find} in {stressed}"
            instr = f"Say the following with the same voice: '{body}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=args.seed)
            out = os.path.join(args.out, f"r2p{pi}_r{len(manifest)}.{name}.wav")
            save_audio(audio, sr, out)
            manifest.append({"phrase": pi, "variant": name, "text": text, "body": body,
                             "gen": out, "ref": REF, "gen_seconds": secs, "seed": args.seed})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("RESPELL2_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
