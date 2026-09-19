# BLIND_V3_FULL: полный слепой прогон Gemini-судьи v3 (tts+tool+capability) — 17.09.2026

Прогон: `audit_gemini_judge_v3.py --blind --areas tts,tool,capability`,
модель `gemini-3.8-flash-medium` (мост 127.0.0.1:8045), слепой режим (LISTEN.csv, SECRET_map.csv
не открывался). Артефакт: `results_audit_gemini_v3_blind_rest.jsonl` (+ `.raw.jsonl`).
Фонетика ранее оценена отдельно (80/80 ok) → `PHONETICS_GEMINI_GIGAAM.md`.

## Итоги по областям (все 5 вариантов вместе; разбивка по вариантам — после раскрытия карты)

| Область | ok | invalid_input | mean overall | годен / доработка / брак |
|---|---:|---:|---:|---|
| tts (240) | 240 | 0 | 5.41 | 80 / 76 / 84 |
| tool (175) | 175 | 0 | 7.49 | 120 / 6 / 49 |
| capability (55) | 55 | 0 | **2.76** | 4 / 8 / 43 |
| фонетика (80) | 80 | 0 | 7.36 | 62 / 27 / 1 |

5 файлов `invalid_input` — все `cap_vocal_extraction`: речевой референс не годится для
выделения вокала (физически неприменимо), НЕ брак модели.

## Выводы (сигнал судьи, не решение)

1. **capability-арсенал почти не работает** у всех вариантов (mean 2.76; 43/55 «брак»):
   эмоции, шёпот, deaccent, enhancement, speech_edit и пр. — согласуется с COMPARISON.md
   (overall 2.3-3.0). Это главный резерв следующей фазы.
2. **tools**: судья считает volume/pitch выполненными (10/10) — но DSP-замеры
   (`TOOL_DSP.md`) уже доказали обратное по величине: volume +1.4-2 дБ при цели +6,
   pitch +3.7-4.3 st при +2. Ещё раз: magnitude по LLM не меряется. Худшие по судье —
   `tool_noise_add` (denoise: 0.4-8.4, разброс) и `tool_speed_up` (2.4-8.2) —
   согласуется с DSP: denoise FAIL у всех, speed — PASS только у B.
3. **tts**: mean 5.41, треть «брак» — в основном фразы с ж/з, ч/ц, ы/и (фонемные замены
   подтверждены GigaAM в `PHONETICS_GEMINI_GIGAAM.md`: цапля→«сапля», сыты→«зыты»,
   крыльца→«крильца»).

## Сопоставление с абсолютными порогами (GATES §1) — статусы

| Область | Порог | Текущий факт | Статус |
|---|---|---|---|
| TTS WER | ≤0.20 | B@500 0.19 | PASS (только B@500) |
| TTS first_ok | ≥0.70 | 0.667 | FAIL |
| Клон sim | ≥0.75 | 0.726-0.777 | FAIL |
| Клон WER | ≤0.15 | 0.35-0.42 | FAIL |
| volume | ≤2 дБ | +1.4…2.0 при +6 | FAIL |
| pitch | ≤1 st | ±3.7-4.3 при ±2 | FAIL |
| denoise | ≥+3 дБ | −3.6…−5.8 | FAIL |
| фонемные замены | ≤0.10 | не измерено надёжно | ИЗМЕРЕНО ТЕПЕРЬ (см. отчёт фонетики) |

## Следующий шаг

1. Speaker-disjoint tools val + DSP-цели volume/pitch/denoise (`tools_overlap.py`).
2. Upstream-контроль по `original_eval_manifest.json` (отделить «не умела» от «сломали»).
3. s3 от B@500 (команда в `GATES_AND_INTERVENTION.md` §4), с тегами ц/с, ь, з/с на
   проблемных словах, найденных Gemini+GigaAM.
