"""Targeted A/B for the word 'развитие' failure: 3 text representations x 2 seeds on a given checkpoint."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instruct", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(AUK, "local_tests", "razvitie_ab", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.ru_translit import ru_to_latin

    row = json.loads(open(os.path.join(AUK, "local_train", "data", "val.jsonl"), encoding="utf-8").readline())
    full = row["messages"][0]["content"][0]["text"]
    stressed = full.split("'", 2)[1]
    plain = stressed.replace("+", "").replace("'", "")
    reps = {"cyr_stress": stressed, "cyr_plain": plain, "translit": ru_to_latin(stressed)}
    ref = os.path.join(AUK, "local_tests", "user_ref.wav")

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    reps_sel = [("cyr_stress", stressed), ("cyr_plain", plain)] if args.instruct else list(reps.items())
    for rep, text in reps_sel:
        for seed in [1234, 7]:
            if args.instruct:
                instr = f"Say the following in Russian with clear, natural pronunciation: '{text}'"
                content = [{"type": "text", "text": instr}]
                ref_arg = None
            else:
                instr = f"Say the following with the same voice: '{text}'"
                content = [{"type": "text", "text": instr}, {"type": "audio", "audio": ref}]
                ref_arg = ref
            messages = [{"role": "user", "content": content}]
            audio, sr = engine.generate(messages, audio=ref_arg, gen_seconds=5.5, nfe=32,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(out_dir, f"razv_{rep}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": len(manifest), "text": plain, "ref": ref, "gen": out,
                             "gen_seconds": 5.5, "representation": rep, "seed": seed})
            print(f"generated {out}", flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("RAZV_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
