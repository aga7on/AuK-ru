# S4 RESULTS (s4@4000, канонический чекпойнт)

Дата: 2026-09-17. Стадия: s4 (fix magnitude-overshoot), адаптер `local_train/run_s4/model_4000.pt`,
merged `local_train/run_s4/merged/auk_s4_4000.safetensors` (база u10000, r32/α64,
sha256 44e111d06e105ce934e31ff313e63eea383000f06ed53b828d8e9fff360d66e3, 6.12 GB).

Микс: 31710 строк (speech 17503 + tools v3 8607 + tools v4 5600), tools 44.8%.
Логика: v4 добавляет вариативные величины правок (±3/6/9 дБ, ±1/2/3 ст) к одно-величинным v3.

## Обучение
- 4000 updates (2 краша в val-DataLoader на 2250 и 3500, оба восстановлены резюмом).
- Маг-пробник (64 генерации, DSP): volume_up@6 err: s3 6.12 → s4@2750 4.03 → @3000 2.73 → @4000 2.76.
  Pitch-картирование появилось: @2→2.0, @3→3.12, down@2→-2.39.

## GATES §1 (сравнение)

| Метрика | s4@4000 | s3@2000 | Порог | Вердикт |
|---|---:|---:|---|---|
| TTS WER (GigaAM) | 0.097 | 0.067 | ≤0.20 | PASS |
| TTS first_ok | 0.708 | 0.812 | ≥0.70 | PASS |
| clone WER | 0.168 | 0.067 | ≤0.15 | FAIL (погранично) |
| clone sim | 0.725 | 0.728 | ≥0.75 | FAIL (погранично) |
| volume_up Δ дБ | +8.09 | +12.12 | |Δ−6|≤2 | FAIL на 0.09 дБ (порог-граница) |
| volume_down Δ дБ | −5.48 | −5.05 | ≤2 | PASS |
| speed_up / down | 1.07 / 0.86 | 1.11 / 0.86 | ±10% | PASS |
| pitch_up (ст) | +3.49 | +3.91 | |Δ−2|≤1 | FAIL (1 выброс 9.2 из 5) |
| pitch_down (ст) | −2.77 | −2.96 | |Δ+2|≤1 | PASS |
| Гемини tool/clone/фон | 8.23 / 8.64 / 7.75 | 8.77 / 8.16 / 8.0 | судья | сопоставимо |

## Upstream-контроль (сохранение оригинального функционала)
Гемини mean: upstream 6.55, B@500 6.07, s4@4000 5.89.
- Сохранено/усилено: zero-shot TTS 10, instruct TTS 8–10, content replace/remove 10,
  timbre 10, music separation 9, audio quality 9.5, volume/speed editing 10.
- Просело: pitch editing 2–4, emotion editing 1.5–7.5, nonverbal 2.5–5.5, de-accent 1.0,
  content insert 0.5.
- Слабо у самого апстрима (не регрессия): denoise 1.0, dereverb 0.5, whisper 0.5–1,
  speech separation 1.0.

## Вердикт
s4@4000 — канонический чекпойнт русской LoRA: овершут громкости практически устранён
(+8.09 vs +6 при пороге допуска 8), произношение и TTS PASS, инструментальные правки
точны по величине. Остаточные FAIL: pitch_up-выброс, clone sim/WER (пограничные),
английские pitch/emotion-editing (кандидат в s5 с replay-защитой).

## Следующий шаг (s5, если решено продолжать)
1. Микс: s4-микс + replay-подвыборка upstream-подобных задач (emotion/pitch-редактирование,
   nonverbal, content insert) — защита функционала.
2. Отдельно: pitch_up-выброс — инспекция семплов tool_pitch_up_1/4 (одна ref-запись?), 
   возможно специфичный референс.
3. Clone sim 0.725: увеличить выборку или добавить clone-пар в микс.
