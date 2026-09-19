"""Генерация replay-набора для s5: учитель = auk_base (upstream), задачи оригинального
функционала на фикстурах demo-input-audio, вариативные величины правок.

Выход: local_train/data_s2_full/v5_replay/replay.jsonl (rows как в train.jsonl: messages+duration)
+ аудио-цели в local_train/data_s2_full/v5_replay/wav/.
Только ретекстур задач, где upstream силён (>=7 по upstream-судье): pitch, emotion, nonverbal,
content-edit, timbre, TTS. Enhancement/separation исключены (слабы у самого upstream).
"""
import json
import os
import random
import sys
import time

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

OUT = os.path.join(AUK, "local_train", "data_s2_full", "v5_replay")
WAV = os.path.join(OUT, "wav")
TEACHER = os.path.join(AUK, "ckpts", "AuK", "auk_base.safetensors")
TEACHER_CFG = os.path.join(AUK, "ckpts", "AuK", "config.yaml")
SEED = 7

# (name, ref, gen_seconds, instruction_templates)
FIXTURES = [
    ("pitch", "assets/demo-input-audio/pitch/pitch-1-input.wav", 5.5,
     ["Raise the pitch by {m} semitones.", "Lower the pitch by {m} semitones."]),
    ("emotion", "assets/demo-input-audio/emotion-edit/en-1-input.wav", 6.6,
     ["Change the emotion to {e}."]),
    ("nonverbal", "assets/demo-input-audio/nv/en-c-input.wav", 10.0,
     ["Add a breath before \"{w}\".", "Remove the humming from the audio.",
      "Add a short laugh after \"{w}\"."]),
    ("content", "assets/demo-input-audio/content-edit/content.wav", 7.0,
     ["Replace 'but accepting what we cannot have' with 'and living with what we can'.",
      "Add 'and here' after 'we cannot have'.", "Remove 'cannot'."]),
    ("timbre", "assets/demo-input-audio/vc/vc-1-input.wav", 6.6,
     ["Keep the spoken content unchanged and change the timbre to: a warm middle-aged male voice.",
      "Keep the spoken content unchanged and change the timbre to: a young female voice."]),
    ("zstts", "assets/demo-input-audio/zero-shot-tts/ref.wav", 6.0,
     ["Say the following with the same voice: '{t}'"]),
]

PITCH_MS = [1, 2, 3]
EMOTIONS = ["happy", "sad", "angry", "excited", "calm"]
ZSTTS_TEXTS = [
    "Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience.",
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "Every journey begins with a single step, and ours started on a cold winter morning.",
]
NONVERBAL_WORDS = ["We tested", "the model", "our results"]

RNG = random.Random(SEED)


def build_tasks():
    tasks = []
    for name, ref, gs, templates in FIXTURES:
        refp = os.path.join(AUK, ref)
        for t in templates:
            if "{m}" in t:
                for m in PITCH_MS:
                    tasks.append({"fixture": name, "ref": refp, "gen_seconds": gs,
                                  "instruction": t.format(m=m)})
            elif "{e}" in t:
                for e in EMOTIONS:
                    tasks.append({"fixture": name, "ref": refp, "gen_seconds": gs,
                                  "instruction": t.format(e=e)})
            elif "{t}" in t:
                for txt in ZSTTS_TEXTS:
                    tasks.append({"fixture": name, "ref": refp, "gen_seconds": max(gs, len(txt) / 14.0),
                                  "instruction": t.format(t=txt)})
            elif "{w}" in t:
                for w in NONVERBAL_WORDS:
                    tasks.append({"fixture": name, "ref": refp, "gen_seconds": gs,
                                  "instruction": t.format(w=w)})
            else:
                tasks.append({"fixture": name, "ref": refp, "gen_seconds": gs, "instruction": t})
    return tasks


def main():
    os.makedirs(WAV, exist_ok=True)
    tasks = build_tasks()
    print(f"tasks={len(tasks)}", flush=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    engine = AukInfer(config_path=TEACHER_CFG, ckpt_path=TEACHER,
                      qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                      cpu_offload=True, device="cuda:1", dtype="bf16")

    rows = []
    t0 = time.time()
    for i, tk in enumerate(tasks):
        out = os.path.join(WAV, f"replay_{i:03d}.wav")
        if os.path.exists(out):
            status = "cached"
        else:
            try:
                content = [{"type": "text", "text": tk["instruction"]},
                           {"type": "audio", "audio": tk["ref"]}]
                messages = [{"role": "user", "content": content}]
                audio, sr = engine.generate(messages, audio=tk["ref"],
                                            gen_seconds=float(tk["gen_seconds"]),
                                            nfe=64, cfg_strength=2.0, seed=SEED)
                save_audio(audio, sr, out)
                status = "ok"
            except Exception as ex:
                status = f"error {type(ex).__name__}"
        if status == "ok" or status == "cached":
            import soundfile as sf
            dur = sf.info(out).duration
            rows.append({"fixture": tk["fixture"], "instruction": tk["instruction"],
                         "ref": tk["ref"], "target": out, "duration": round(dur, 2)})
        print(f"[{i+1}/{len(tasks)}] {tk['fixture']} {status} ({(time.time()-t0)/60:.1f}m)", flush=True)

    json.dump(rows, open(os.path.join(OUT, "replay_tasks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"REPLAY_DONE ok={len(rows)}/{len(tasks)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
