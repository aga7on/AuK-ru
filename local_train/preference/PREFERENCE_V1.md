# PREFERENCE v1 — automatic candidate mining (S10 prep)

Пар: **56** (missing files: 0 — должно быть 0)

| источник | пар | метрика |
|---|---|---|
| tmp_seed_probe_s7_5750 | 23 | wespeaker_sim |
| emotion_ru_s7_5750 | 17 | judge_overall |
| emotion_ru_s7_6750 | 16 | judge_overall |

Правила: sim-пары gap ≥ 0.03; judge-пары — различие verdict или Δoverall ≥ 2.
Назначение: seed для S10 preference/rejection-обучения (first-shot reliability).
Расширение: N>1 майнинг на clone100-промптах (4–8 кандидатов) — следующий шаг S10.
