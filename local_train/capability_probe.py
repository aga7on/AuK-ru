"""Capability probe on merged u18000: editing/paralinguistic tools from base AuK.

Our s1 LoRA was trained on TTS-only data; these tasks run through the merged base
capability — this probe checks they still work (GigaAM transcript + files for the ear).
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")
CKPT = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
OUT = os.path.join(AUK, "local_tests", "capability_probe")

CASES = [
    ("emotion_happy", "Say this in a happy tone"),
    ("emotion_sad", "Say this in a sad tone"),
    ("emotion_angry", "Say this in an angry tone"),
    ("whisper", "Turn this into a whisper"),
    ("speech_edit", "Replace '\u043f\u0440\u0438\u0432\u0435\u0442' with '\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435'."),
    ("deaccent", "Remove any accent from this speech, keep the same voice."),
    ("enhancement", "Remove the background noise and make the voice cleaner"),
    ("quality", "Improve the audio quality and make it clearer"),
    ("speed_12", "Change the speech speed to 1.2 times."),
    ("pitch_up", "Raise the pitch by 2 semitones."),
    ("nonverbal_breath", "Add a breath before the last word"),
    ("vocal_extraction", "Extract the vocals and remove the accompaniment"),
]

from auk.infer.infer_auk import AukInfer, save_audio  # noqa: E402
from auk.infer import quality  # noqa: E402

if __name__ == "__main__":
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(CKPT), "config.yaml"),
        ckpt_path=CKPT,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for key, instr in CASES:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": REF},
        ]}]
        try:
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=10.21, nfe=64,
                                        cfg_strength=2.0, seed=1234)
            audio = quality.trim_silence(audio, sr)
            out = os.path.join(OUT, f"cap_{key}.wav")
            save_audio(audio, sr, out)
            heard = quality.transcribe(audio, sr)
            manifest.append({"case": key, "instruction": instr, "file": out, "asr_heard": heard})
            print(f"{key}: ok | heard: {heard[:70]}", flush=True)
        except Exception as e:
            manifest.append({"case": key, "instruction": instr, "error": f"{type(e).__name__}: {e}"})
            print(f"{key}: FAILED {type(e).__name__}", flush=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("CAPABILITY_PROBE_DONE")
