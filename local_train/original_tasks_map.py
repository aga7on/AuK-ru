"""Inventory the ORIGINAL AuK task catalog and build a reviewable eval manifest.

Source of truth: docs/COOKBOOK.md + the first (full) DEMO_EXAMPLE_GROUPS in
src/auk/infer/infer_gradio.py, with fixtures under assets/demo-input-audio.
No synthesis is run; this only prepares a manifest for a later upstream-control benchmark.
"""
import json
import os

AUK = r"G:\AI\AuK"
OUTDIR = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
DEMO = "assets/demo-input-audio"

# (group, task, instruction, fixture (or None), gen_seconds, notes)
CATALOG = [
    ("Speech Generation", "Zero-shot TTS", "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'", f"{DEMO}/zero-shot-tts/ref.wav", 6.0, "EN; ref voice"),
    ("Speech Generation", "Instruct TTS", "Generate speech based on the following description: \"一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断\", and say: \"宁可我负天下人，休教天下人负我。\"", None, 3.36, "no ref audio; text-only"),
    ("Speech Generation", "Instruct TTS", "Generate speech based on the following description: \"一位中年女性，用沙哑、干涩的嗓音倾诉，语速缓慢而沉稳\", and say: \"我的眼泪早哭干了，我没有委屈，我有的是恨。\"", None, 15.0, "no ref audio; text-only"),
    ("Content Editing", "Speech Content Editing (replace)", "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.", f"{DEMO}/content-edit/content.wav", 7.0, "verify source contains the original span"),
    ("Content Editing", "Speech Content Editing (insert)", "Add 'and here' after 'we cannot have'.", f"{DEMO}/content-edit/content.wav", 7.0, "cookbook template"),
    ("Content Editing", "Speech Content Editing (remove)", "Remove 'cannot'.", f"{DEMO}/content-edit/content.wav", 7.0, "cookbook template"),
    ("Content Editing", "Lyric Editing", 'Change "rear view" to "like you" in the vocal recording.', f"{DEMO}/vocal-edit/vocaledit-en-1-input.wav", 5.44, "needs a cappella vocal"),
    ("Acoustic Editing", "Pitch Editing", "Raise the pitch by 2 semitones.", f"{DEMO}/pitch/pitch-1-input.wav", 5.12, "objective: +2 st"),
    ("Acoustic Editing", "Pitch Editing", "Lower the pitch by 2 semitones.", f"{DEMO}/pitch/pitch-1-input.wav", 5.0, "objective: -2 st"),
    ("Acoustic Editing", "Speed Editing", "Adjust the speech speed to 0.75x.", f"{DEMO}/speed/speed-edit-1-input.wav", 13.74, "objective: rate 0.75"),
    ("Acoustic Editing", "Speed Editing", "Adjust the speech speed to 1.5x.", f"{DEMO}/speed/speed-edit-1-input.wav", 6.86, "objective: rate 1.5"),
    ("Acoustic Editing", "Volume Editing", "Increase the volume by 10 dB.", f"{DEMO}/energy/energy-edit-1-input.wav", 3.78, "objective: +10 dB"),
    ("Acoustic Editing", "Volume Editing", "Decrease the volume by 10 dB.", f"{DEMO}/energy/energy-edit-1-input.wav", 3.78, "objective: -10 dB"),
    ("Paralinguistic Editing", "Emotion Editing", "Change the emotion to happy.", f"{DEMO}/emotion-edit/en-1-input.wav", 6.6, "cookbook template"),
    ("Paralinguistic Editing", "Emotion Editing", "Change the emotion to angry.", f"{DEMO}/emotion-edit/en-1-input.wav", 6.6, "cookbook template"),
    ("Paralinguistic Editing", "Emotion Editing", "Change the emotion to sad.", f"{DEMO}/emotion-edit/en-1-input.wav", 7.6, "cookbook template"),
    ("Paralinguistic Editing", "Timbre Editing", "Keep the spoken content unchanged and change the timbre to: \"a deep, calm male voice\".", f"{DEMO}/vc/vc-1-input.wav", 6.56, "timbre change"),
    ("Paralinguistic Editing", "De-accent", "Remove the regional accent while preserving the speaker's voice and content.", f"{DEMO}/accent/accent-sichuan-input.wav", 6.74, "accented input"),
    ("Paralinguistic Editing", "Nonverbal Editing (remove)", "Remove the humming from the audio.", f"{DEMO}/nv/en-d-input.wav", 22.0, "removal"),
    ("Paralinguistic Editing", "Nonverbal Editing (add)", "Add a breath before \"We tested\"", f"{DEMO}/nv/en-c-input.wav", 10.44, "additive"),
    ("Paralinguistic Editing", "Whisper Conversion (to whisper)", "Convert this speech into a soft whisper while preserving the speaker and content.", f"{DEMO}/whisper/wh-w2n-zh-input.wav", 8.36, "to-whisper"),
    ("Paralinguistic Editing", "Whisper Conversion (from whisper)", "Convert this whispered speech into a normal speaking voice while preserving the speaker and content.", f"{DEMO}/whisper/wh-w2n-zh-input.wav", 8.36, "from-whisper, needs whispered input fixture"),
    ("Enhancement & Separation", "Speech Enhancement (denoise)", "Remove only the background noise, preserve everything else, and output audio of the same length.", f"{DEMO}/se/se-zh-1-input.wav", 0, "same length"),
    ("Enhancement & Separation", "Speech Enhancement (dereverberate)", "Remove only the room reverberation, preserve everything else, and output audio of the same length.", f"{DEMO}/se/se-zh-1-input.wav", 0, "needs reverberant fixture"),
    ("Enhancement & Separation", "Speech Enhancement (full)", "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.", f"{DEMO}/se/se-zh-1-input.wav", 0, "same length"),
    ("Enhancement & Separation", "Speech Separation (by order)", "Keep only the first speaker.", f"{DEMO}/ss/en-1-input.wav", 28.0, "multi-speaker mix"),
    ("Enhancement & Separation", "Speech Separation (by order)", "Keep only the second speaker", f"{DEMO}/ss/zh-1-input.wav", 18.24, "multi-speaker mix"),
    ("Enhancement & Separation", "Target Speaker Extraction", 'Keep only the speaker who says "get what" and remove all other speakers.', f"{DEMO}/ss/en-1-input.wav", 28.0, "by content"),
    ("Enhancement & Separation", "Music Separation (vocals)", "Extract the vocals and remove the accompaniment", f"{DEMO}/vocal-extraction/vocal-1-input.wav", 10.88, "must be music+vocal mix"),
    ("Enhancement & Separation", "Music Separation (all voices)", "Keep all human voices, including speech and singing, and remove everything else.", f"{DEMO}/vocal-extraction/vocal-1-input.wav", 10.88, "cookbook template"),
    ("Enhancement & Separation", "Audio Quality Enhancement", "Improve the audio quality and make it clearer", f"{DEMO}/se/se-zh-1-input.wav", 5.0, "quality"),
]

# Current Russian pack coverage (id -> original task family)
COVERED = {
    "zero_shot_tts_ru": "Speech Generation / Zero-shot TTS (RU)",
    "emotion_happy": "emotion", "emotion_sad": "emotion", "emotion_angry": "emotion",
    "whisper": "whisper to-whisper",
    "speech_edit_replace": "content editing replace (RU demo)",
    "deaccent": "de-accent", "enhancement": "enhancement denoise", "quality": "quality",
    "speed_12": "speed (1.2x)", "pitch_up": "pitch (+2st)", "nonverbal_breath": "nonverbal add breath",
    "vocal_extraction": "music separation vocals (INVALID: speech-only ref)",
}
MISSING = [
    "Instruct TTS (no-ref) — not in pack",
    "Timbre editing (vc) — not in pack",
    "Lyric editing (vocal-edit) — not in pack",
    "Speech separation by order / target-speaker extraction — not in pack",
    "Dereverberate / full-enhancement distinction — not in pack",
    "Nonverbal removal (breath/humming) — only addition in pack",
    "Whisper from-whisper — only to-whisper in pack",
    "Emotions fear/surprise/disgust/calm/excited — only happy/sad/angry in pack",
    "Volume/speed/pitch magnitude sweeps (multi-point) — single point in pack",
    "Zero-shot TTS in English on upstream fixtures — not covered",
]


def main():
    entries, missing_fixture = [], []
    for group, task, instr, fixture, gen, note in CATALOG:
        path = os.path.join(AUK, fixture) if fixture else None
        exists = bool(path and os.path.exists(path))
        if fixture and not exists:
            missing_fixture.append(fixture)
        entries.append({"group": group, "task": task, "instruction": instr,
                        "audio": path, "gen_seconds": gen, "fixture_exists": exists,
                        "notes": note})
    manifest = {
        "purpose": "Original-task (upstream) control benchmark. Run later against base u10000 and s2; no synthesis now.",
        "checkpoint_variants": {
            "upstream": r"G:\AI\AuK\ckpts\AuK\auk_base.safetensors",
            "s0": r"G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors",
            "s1_u10000": r"G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors",
            "s2_A500": r"G:\AI\AuK\local_train\run_s2_A\merged\auk_s2_A_500.safetensors",
            "s2_B500": r"G:\AI\AuK\local_train\run_s2_B\merged\auk_s2_B_500.safetensors",
        },
        "entries": entries,
    }
    json.dump(manifest, open(os.path.join(OUTDIR, "original_eval_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    L = ["# Инвентаризация оригинального каталога AuK (16.09.2026)", "",
         "Источник: `docs/COOKBOOK.md` и первый (полный) `DEMO_EXAMPLE_GROUPS` в `src/auk/infer/infer_gradio.py`;",
         "фикстуры — `assets/demo-input-audio/`. Оценка не запускается; подготовлен reviewable-manifest", "",
         "| группа | задача | вход | фикстура | есть | проверяет |",
         "|---|---|---|---|---:|---|"]
    for e in entries:
        rel = os.path.relpath(e["audio"], AUK) if e["audio"] else "— (text-only)"
        L.append(f"| {e['group']} | {e['task']} | {'audio' if e['audio'] else 'none'} | {rel} | "
                 f"{'да' if e['fixture_exists'] else '—'} | {e['notes']} |")
    L += ["", "## Покрытие текущим русским pack", "",
          "| оригинальная семья | покрыто в pack |", "|---|---|"]
    for k, v in COVERED.items():
        L.append(f"| {v} | {k} |")
    L += ["", "## Чего нет в текущем pack", ""]
    for m in MISSING:
        L.append(f"- {m}")
    L += ["", "## Известные проблемы валидности фикстур", "",
          "- `speech_edit` (RU демо) требует, чтобы во входе было слово «привет»; используемый `user_ref.wav`",
          "  этого не гарантирует — тест может быть неприменим.",
          "- `vocal_extraction` с чисто речевым референсом физически некорректен: нужен микс вокал+аккомпанемент",
          "  (`assets/demo-input-audio/vocal-extraction/vocal-1-input.wav`).",
          "- `emotion/whisper/nonverbal` в pack используют кастомные формулировки, а не точные шаблоны оригинального каталога.",
          "- Инструктированный TTS без референса в pack отсутствует; оригинал использует `None` audio.",
          "", f"Фикстур не найдено: {missing_fixture if missing_fixture else 'нет'}", ""]
    open(os.path.join(OUTDIR, "ORIGINAL_TASKS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written ORIGINAL_TASKS.md and original_eval_manifest.json; entries", len(entries),
          "missing fixtures", missing_fixture)


if __name__ == "__main__":
    main()
