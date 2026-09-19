"""Проверка гипотезы: auk_ru_best (s0) — транслит-модель. Кириллица → мусор, транслит → ок."""
import json
import os
import sys

import numpy as np

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

INIT = os.path.join(AUK, "local_train", "run_ru", "auk_ru_best.safetensors")
MERGED_10K = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_10000.safetensors")
OUT = os.path.join(AUK, "local_train", "diagnostics")


def main():
    from auk.infer.infer_auk import AukInfer, save_audio
    from gigaam_asr import transcribe
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer.ru_translit import ru_to_latin

    ref = r"G:\AI\kyutai-ru\data\ru_wav_mfa\e1c4c07bc9e43175.wav"
    text = "сбер. что для мальцева значит дружба?"

    def gen_and_asr(engine, tag, instruction, secs=4.6):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=1234)
        path = os.path.join(OUT, f"hyp_{tag}.wav")
        save_audio(audio, sr, path)
        heard = transcribe(path)
        print(f"  [{tag}]: {heard[:90]}", flush=True)
        return heard

    stressed_cyr = _accentize_ru(text)
    stressed_lat = ru_to_latin(stressed_cyr)

    print("=== auk_ru_best (S0, TRANSLIT-модель) ===")
    e0 = AukInfer(
        config_path=os.path.join(AUK, "local_train", "run_ru", "config.yaml"),
        ckpt_path=INIT, qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True, device="cuda:0", dtype="bf16")
    print(" cyrillic:")
    gen_and_asr(e0, "s0_cyr", f"Say the following with the same voice: '{stressed_cyr}'")
    print(" translit:")
    gen_and_asr(e0, "s0_lat", f"Say the following with the same voice: '{stressed_lat}'")

    print("=== u10000 (S1, CYRILLIC-модель) ===")
    e1 = AukInfer(
        config_path=os.path.join(AUK, "local_train", "run_ru_s1", "merged", "config.yaml"),
        ckpt_path=MERGED_10K, qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True, device="cuda:0", dtype="bf16")
    print(" cyrillic:")
    gen_and_asr(e1, "u10k_cyr", f"Say the following with the same voice: '{stressed_cyr}'")
    print(" translit:")
    gen_and_asr(e1, "u10k_lat", f"Say the following with the same voice: '{stressed_lat}'")

    print("HYPOTHESIS_TEST_DONE")


if __name__ == "__main__":
    main()
