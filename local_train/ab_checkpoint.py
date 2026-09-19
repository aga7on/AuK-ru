"""A/B release candidates: u18000 (user-confirmed) vs u22000 (last, highest rubric).

10 regression phrases x 2 checkpoints, seed 1234, trim + GigaAM screen in-line.
Auto-judge (Gemini, votes=2) runs afterwards on the same dir.
"""
import argparse
import gc
import json
import os
import subprocess
import sys

import torch

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")

PHRASES = [
    "Ещё более важную роль на Африканском Роге играет устойчивое развитие.",
    "Привет! Это проверка русского произношения. Раз, два, три.",
    "Пожалуйста, не опаздывайте на встречу, электронное объявление уже на сайте.",
    "Иван Петров объяснил правила пользования сервисом с двумя тысячами участников.",
    "Взгляд её был спокоен, тёплый вечер принёс запах моря и шелест волн.",
    "Позвони мне, пожалуйста, когда освободишься.",
    "Мне нравится смотреть на звёзды летними ночами.",
    "Две тысячи двадцать шестого года мы запустили этот проект.",
    "Сельдь под шубой — традиционное новогоднее блюдо.",
    "Учитель физики объяснял законы сохранения энергии.",
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.08 * len(text)))


def gen_for_ckpt(ckpt, tag, out, seed=1234):
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer import quality

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(ckpt), "config.yaml"),
        ckpt_path=ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    for pi, text in enumerate(PHRASES):
        body = _accentize_ru(text)
        instr = f"Say the following with the same voice: '{body}'"
        secs = estimate_seconds(text)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": REF},
        ]}]
        audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=seed)
        audio = quality.trim_silence(audio, sr)
        out_path = os.path.join(out, f"{tag}_p{pi:02d}.wav")
        save_audio(audio, sr, out_path)
        heard = quality.transcribe(audio, sr)
        rec = quality.recall(text, heard)
        manifest.append({"tag": tag, "phrase_idx": pi, "text": text, "file": out_path,
                         "recall": round(rec, 3), "asr_heard": heard})
        print(f"{tag} p{pi:02d}: recall={rec:.2f} | {heard[:50]}", flush=True)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def ensure_merged(run_dir, up):
    merged = os.path.join(run_dir, "merged", f"auk_ru_{up}.safetensors")
    if os.path.exists(merged):
        return merged
    pt = os.path.join(run_dir, f"model_{up}.pt")
    if not os.path.exists(pt):
        pt = os.path.join(run_dir, "model_last.pt")
    print(f"merging {os.path.basename(pt)} -> {os.path.basename(merged)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(AUK, "local_train", "merge_lora.py"),
                        "--run_dir", run_dir, "--ckpt", pt, "--out", merged,
                        "--base_ckpt", os.path.join(AUK, "local_train", "run_ru", "auk_ru_best.safetensors"),
                        "--lora_r", "32", "--lora_alpha", "64"], cwd=AUK)
    if r.returncode != 0 or not os.path.exists(merged):
        print(f"MERGE FAILED for u{up}", flush=True)
        return None
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.path.join(AUK, "local_train", "run_ru_s1"))
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "ab_final"))
    ap.add_argument("--updates", default="18000,22000")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for up in [int(x) for x in args.updates.split(",")]:
        ckpt = ensure_merged(args.run_dir, up)
        if not ckpt:
            continue
        rows = gen_for_ckpt(ckpt, f"u{up}", args.out, args.seed)
        all_rows.extend(rows)
    json.dump(all_rows, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("AB_FINAL_GEN_DONE")


if __name__ == "__main__":
    main()
