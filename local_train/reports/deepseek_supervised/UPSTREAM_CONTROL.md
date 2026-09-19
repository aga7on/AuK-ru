# UPSTREAM_CONTROL: оригинальный AuK vs s2 B@500 на 31 задаче оригинала (17.09.2026)

Назначение: отделить «модель не умела» от «русская адаптация сломала».

Прогон: `upstream_control_gen.py` (31 entry × 2 варианта, seed 1234, NFE 64, CFG 2.0,
без trim; cuda:1) → `local_tests/upstream_control/{upstream,B500}` — 31/31 ok у обоих.
Судья: `audit_gemini_judge_v3.py --manifest upstream_judge_manifest.json` →
`results_upstream_judge.jsonl` — 60 ok / 2 no_result (audio_access=false, обе e22/e24
enhancement у B500 — судья не услышал аудио, НЕ брак).

## Свод

| Вариант | n ok | mean overall | mean operation_performed | годен |
|---|---:|---:|---:|---:|
| upstream base | 31 | 6.55 | 6.81 | 18/31 |
| s2 B@500 | 29 | 6.07 | 6.03 | 16/29 |

## Парные регрессии B@500 vs upstream (Δ ≤ −3)

| Задача | upstream | B@500 | Δ |
|---|---:|---:|---:|
| e07 pitch_editing | 10 | 3 | −7 |
| e13 emotion_editing | 8 | 2 | −6 |
| e15 emotion_editing | 8 | 2 | −6 |
| e19 nonverbal_add | 4 | 1 | −3 |
| e30 audio_quality_enhancement | 9 | 1 | −8 |

Полная таблица: `UPSTREAM_COMPARE.tsv`.

## Выводы

1. **Русская адаптация заметно сломала часть арсенала оригинала**: pitch- и
   emotion-editing, nonverbal, audio quality — у upstream работают (8-10), у B@500 — нет
   (1-3). Это согласуется с capability mean 2.76 в слепом прогоне (`BLIND_V3_FULL.md`)
   и с DSP-FAIL pitch.
2. **Что НЕ сломано** (upstream тоже слаб или B сопоставим): separation, target-speaker
   extraction, denoise — парной регрессии нет; часть задач у upstream сама низкая.
3. Следствие для s3: в tools-микс добавить pitch-пар (сейчас недопредставлены),
   и рассмотреть capability-эмоции в трейн-миксе (пункт для следующего протокола).
