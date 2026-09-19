"""Voice benchmark on checkpoint u18000 across 4 reference speakers."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REFS = {
    "user_male": os.path.join(AUK, "local_tests", "user_ref.wav"),
    "nat_female": os.path.join(AUK, "local_tests", "ref_ru.wav"),
    "alt_male": os.path.join(AUK, "local_tests", "ref_male2.wav"),
    "alt_female": os.path.join(AUK, "local_tests", "ref_female2.wav"),
}

PHRASES = [
    ("val_razvitie", "Ещё более важную роль на Африканском Роге играет устойчивое развитие."),
    ("proiznoshenie", "Привет! Это проверка русского произношения. Раз, два, три."),
    ("vstrecha_objavlenie", "Пожалуйста, не опаздывайте на встречу, электронное объявление уже на сайте."),
    ("polzovanie_names", "Иван Петров объяснил правила пользования сервисом с двумя тысячами участников."),
    ("prosody_sea", "Взгляд её был спокоен, тёплый вечер принёс запах моря и шелест волн."),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.08 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors"))
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "voices_u18000"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer import quality

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )

    manifest = []
    for voice_name, ref_path in REFS.items():
        print(f"\n--- Generating voice: {voice_name} ---")
        for p_key, text in PHRASES:
            body = _accentize_ru(text)
            instr = f"Say the following with the same voice: '{body}'"
            secs = estimate_seconds(text)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": ref_path},
            ]}]
            audio, sr = engine.generate(messages, audio=ref_path, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=args.seed)
            # Apply silence trim
            audio_trimmed = quality.trim_silence(audio, sr)
            out_path = os.path.join(args.out, f"{voice_name}_{p_key}.wav")
            save_audio(audio_trimmed, sr, out_path)
            
            # GigaAM ASR verification
            heard = quality.transcribe(audio_trimmed, sr)
            rec = quality.recall(text, heard)
            
            manifest.append({
                "voice": voice_name,
                "phrase_key": p_key,
                "text": text,
                "body": body,
                "ref": ref_path,
                "file": out_path,
                "gen_seconds": secs,
                "asr_heard": heard,
                "recall": round(rec, 3)
            })
            print(f"[{voice_name}] {p_key}: recall={rec:.2f} | {heard[:60]}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("\nVOICES_U18000_DONE")


if __name__ == "__main__":
    main()
