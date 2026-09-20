# PREFERENCE v1 — automatic candidate mining (S10 prep, 19.09.2026)

## Датасет пар

| файл | пар | источник | метрика |
|---|---|---|---|
| `pairs_v1.jsonl` | 56 | seed-проба s7@5750 (23, WeSpeaker sim, gap≥0.03) + эмо-паки s7@5750/s7@6750 (33, judge verdict/Δoverall≥2) | sim / judge |
| `pairs_mined_v1.jsonl` | 18 | s10_candidate_mine: 25 clone100-промптов × 4 seeds (7/123/999/2026) на v1.0, composite gap≥0.05 | composite_v1 |
| `pairs_mined_v2.jsonl` | 37 | s10_mine_v2: 50 HELD-OUT текстов (hard_cases_ru) × 4 seeds × held-out рефы (вне всех train-пулов), composite gap≥0.05 | composite_v1 |

ИТОГО: **111 пар**, missing files = 0 (инвариант проверен).

## First-shot gap (три независимых замера)

| набор | first-seed | best-of-N | gap |
|---|---|---|---|
| seed-проба 25 клонов (sim) | 0.7378 | 0.7711 (best-of-3) | +0.0333 |
| mine v1, 25 clone-промптов (composite) | 0.7526 | 0.7789 (best-of-4) | +0.0263 |
| mine v2, 50 held-out hard-текстов (composite) | 0.7202 | 0.7539 (best-of-4) | +0.0253 |

На held-out hard-текстах gap тот же (+0.025) — феномен «первая попытка хуже лучшей»
не артефакт лёгких текстов. В 18/50 (36%) первый seed уже лучший.

## Composite score (v1, веса ROADMAP S11)

0.35·sim + 0.25·asr_fidelity + 0.20·dnsmos + 0.10·loudness + 0.10·pause − repetition − artifact.

**Ключевой замер S10 (first-shot gap, composite):**
first-seed median **0.7526** vs best-of-4 **0.7789** → gap **+0.0263** (25 промптов, 100 генераций).
Согласуется с sim-only замером (0.7378→0.7711, gap +0.033): модель умеет хороший результат,
но не выдаёт его первой попыткой в ~60% случаев → preference-обучение обосновано.

## Инцидент (записан в ERRORS.MD #4)

`s10_candidate_mine.py` упал на json.dump (np.float32) ПОСЛЕ генерации 99/100 wav —
спас `s10_rescore.py` (пересчёт метрик из wav без перегенерации). Урок: float() при сборке
строк + dry-run сериализации перед тяжёлыми прогонами.

## Следующий шаг S10

- Расширение майнинга: 100 промптов × 4 seeds (300+ пар) после S8-вердикта (на лучшем чекпойнте).
- Формат обучения: пары {prompt, ref, chosen, rejected} → DPO-подобный объект или
  rejection-sampling fine-tune (решается на старте S10).
- Калибровка весов composite — после human A/B (S13).

## Артефакты

- `local_tests/s10_mine/` — 100 wav (25×4) + results.json (sim/asr/nat/loud/pause/rep/art/score)
- `local_train/s10_preference_mine.py`, `local_train/s10_candidate_mine.py`, `local_train/s10_rescore.py`