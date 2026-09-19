# Инвентаризация оригинального каталога AuK (16.09.2026)

Источник: `docs/COOKBOOK.md` и первый (полный) `DEMO_EXAMPLE_GROUPS` в `src/auk/infer/infer_gradio.py`;
фикстуры — `assets/demo-input-audio/`. Оценка не запускается; подготовлен reviewable-manifest

| группа | задача | вход | фикстура | есть | проверяет |
|---|---|---|---|---:|---|
| Speech Generation | Zero-shot TTS | audio | assets\demo-input-audio\zero-shot-tts\ref.wav | да | EN; ref voice |
| Speech Generation | Instruct TTS | none | — (text-only) | — | no ref audio; text-only |
| Speech Generation | Instruct TTS | none | — (text-only) | — | no ref audio; text-only |
| Content Editing | Speech Content Editing (replace) | audio | assets\demo-input-audio\content-edit\content.wav | да | verify source contains the original span |
| Content Editing | Speech Content Editing (insert) | audio | assets\demo-input-audio\content-edit\content.wav | да | cookbook template |
| Content Editing | Speech Content Editing (remove) | audio | assets\demo-input-audio\content-edit\content.wav | да | cookbook template |
| Content Editing | Lyric Editing | audio | assets\demo-input-audio\vocal-edit\vocaledit-en-1-input.wav | да | needs a cappella vocal |
| Acoustic Editing | Pitch Editing | audio | assets\demo-input-audio\pitch\pitch-1-input.wav | да | objective: +2 st |
| Acoustic Editing | Pitch Editing | audio | assets\demo-input-audio\pitch\pitch-1-input.wav | да | objective: -2 st |
| Acoustic Editing | Speed Editing | audio | assets\demo-input-audio\speed\speed-edit-1-input.wav | да | objective: rate 0.75 |
| Acoustic Editing | Speed Editing | audio | assets\demo-input-audio\speed\speed-edit-1-input.wav | да | objective: rate 1.5 |
| Acoustic Editing | Volume Editing | audio | assets\demo-input-audio\energy\energy-edit-1-input.wav | да | objective: +10 dB |
| Acoustic Editing | Volume Editing | audio | assets\demo-input-audio\energy\energy-edit-1-input.wav | да | objective: -10 dB |
| Paralinguistic Editing | Emotion Editing | audio | assets\demo-input-audio\emotion-edit\en-1-input.wav | да | cookbook template |
| Paralinguistic Editing | Emotion Editing | audio | assets\demo-input-audio\emotion-edit\en-1-input.wav | да | cookbook template |
| Paralinguistic Editing | Emotion Editing | audio | assets\demo-input-audio\emotion-edit\en-1-input.wav | да | cookbook template |
| Paralinguistic Editing | Timbre Editing | audio | assets\demo-input-audio\vc\vc-1-input.wav | да | timbre change |
| Paralinguistic Editing | De-accent | audio | assets\demo-input-audio\accent\accent-sichuan-input.wav | да | accented input |
| Paralinguistic Editing | Nonverbal Editing (remove) | audio | assets\demo-input-audio\nv\en-d-input.wav | да | removal |
| Paralinguistic Editing | Nonverbal Editing (add) | audio | assets\demo-input-audio\nv\en-c-input.wav | да | additive |
| Paralinguistic Editing | Whisper Conversion (to whisper) | audio | assets\demo-input-audio\whisper\wh-w2n-zh-input.wav | да | to-whisper |
| Paralinguistic Editing | Whisper Conversion (from whisper) | audio | assets\demo-input-audio\whisper\wh-w2n-zh-input.wav | да | from-whisper, needs whispered input fixture |
| Enhancement & Separation | Speech Enhancement (denoise) | audio | assets\demo-input-audio\se\se-zh-1-input.wav | да | same length |
| Enhancement & Separation | Speech Enhancement (dereverberate) | audio | assets\demo-input-audio\se\se-zh-1-input.wav | да | needs reverberant fixture |
| Enhancement & Separation | Speech Enhancement (full) | audio | assets\demo-input-audio\se\se-zh-1-input.wav | да | same length |
| Enhancement & Separation | Speech Separation (by order) | audio | assets\demo-input-audio\ss\en-1-input.wav | да | multi-speaker mix |
| Enhancement & Separation | Speech Separation (by order) | audio | assets\demo-input-audio\ss\zh-1-input.wav | да | multi-speaker mix |
| Enhancement & Separation | Target Speaker Extraction | audio | assets\demo-input-audio\ss\en-1-input.wav | да | by content |
| Enhancement & Separation | Music Separation (vocals) | audio | assets\demo-input-audio\vocal-extraction\vocal-1-input.wav | да | must be music+vocal mix |
| Enhancement & Separation | Music Separation (all voices) | audio | assets\demo-input-audio\vocal-extraction\vocal-1-input.wav | да | cookbook template |
| Enhancement & Separation | Audio Quality Enhancement | audio | assets\demo-input-audio\se\se-zh-1-input.wav | да | quality |

## Покрытие текущим русским pack

| оригинальная семья | покрыто в pack |
|---|---|
| Speech Generation / Zero-shot TTS (RU) | zero_shot_tts_ru |
| emotion | emotion_happy |
| emotion | emotion_sad |
| emotion | emotion_angry |
| whisper to-whisper | whisper |
| content editing replace (RU demo) | speech_edit_replace |
| de-accent | deaccent |
| enhancement denoise | enhancement |
| quality | quality |
| speed (1.2x) | speed_12 |
| pitch (+2st) | pitch_up |
| nonverbal add breath | nonverbal_breath |
| music separation vocals (INVALID: speech-only ref) | vocal_extraction |

## Чего нет в текущем pack

- Instruct TTS (no-ref) — not in pack
- Timbre editing (vc) — not in pack
- Lyric editing (vocal-edit) — not in pack
- Speech separation by order / target-speaker extraction — not in pack
- Dereverberate / full-enhancement distinction — not in pack
- Nonverbal removal (breath/humming) — only addition in pack
- Whisper from-whisper — only to-whisper in pack
- Emotions fear/surprise/disgust/calm/excited — only happy/sad/angry in pack
- Volume/speed/pitch magnitude sweeps (multi-point) — single point in pack
- Zero-shot TTS in English on upstream fixtures — not covered

## Известные проблемы валидности фикстур

- `speech_edit` (RU демо) требует, чтобы во входе было слово «привет»; используемый `user_ref.wav`
  этого не гарантирует — тест может быть неприменим.
- `vocal_extraction` с чисто речевым референсом физически некорректен: нужен микс вокал+аккомпанемент
  (`assets/demo-input-audio/vocal-extraction/vocal-1-input.wav`).
- `emotion/whisper/nonverbal` в pack используют кастомные формулировки, а не точные шаблоны оригинального каталога.
- Инструктированный TTS без референса в pack отсутствует; оригинал использует `None` audio.

Фикстур не найдено: нет

