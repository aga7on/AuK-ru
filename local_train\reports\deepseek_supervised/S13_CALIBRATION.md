# S13 → S11 CALIBRATION — детектор «жуёт слова» на human-флагах

n=78 файлов, human «жуёт» = 23 (29%).

## Point-biserial корреляции признаков судьи с human-флагом

| признак | corr |
|---|---|
| text_fidelity | -0.583 |
| verdict_brak | 0.544 |
| n_mangled | 0.424 |
| endings | -0.398 |
| overall | -0.378 |
| artifacts | -0.354 |
| palatalization | -0.302 |
| stress | -0.302 |
| accent | -0.262 |
| n_subs | 0.239 |
| naturalness | -0.235 |
| prosody | -0.171 |
| n_dropped | 0.017 |

## Правила-детекторы (precision/recall против human)

| правило | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| verdict==брак | 16 | 8 | 7 | 0.67 | 0.70 |
| text_fidelity<=5 | 8 | 1 | 15 | 0.89 | 0.35 |
| accent<=6 | 8 | 15 | 15 | 0.35 | 0.35 |
| n_mangled>=1 | 20 | 20 | 3 | 0.50 | 0.87 |
| n_subs>=1 | 13 | 17 | 10 | 0.43 | 0.57 |
| overall<=5 | 18 | 25 | 5 | 0.42 | 0.78 |
| composite: fid<=6 or mangled>=1 or subs>=1 | 21 | 21 | 2 | 0.50 | 0.91 |

## Рекомендация для composite reranker (S11)

Выбирается правило с лучшим F1; штраф в composite score за его срабатывание
(текущий rep_pen/art_pen расширяется: mumble_pen = 0.15·(fid≤6) + 0.10·(mangled≥1)
+ 0.10·(subs≥1)). Калибровка приблизительная (n=78, один слушатель) — уточнить
на следующем A/B-наборе.


## Валидация ASR-прокси (validate_mumble_proxy.py, n=80) — ВАЖНО

Реализованы ДВА детектора «жуёт» в rerank_composite.py:
1. `judge_mumble_penalty(judge)` — по осям судьи v3 (text_fidelity/words_mangled/
   phoneme_substitutions/verdict). Калиброванное правило (recall 0.91 / precision 0.50);
   требует Gemini-прокси → для офлайн-гейтов и батч-обработки.
2. `mumble_penalty(expected, heard)` — ASR-прокси (GigaAM) для реалтайм-reranking БЕЗ судьи.
   **Проверен против тех же human-флагов: TP=12 FP=12 FN=12 TN=44 → precision 0.50,
   recall 0.50 — WEAK.** Причина: GigaAM сам ошибается на hard-текстах (числа/даты),
   WER-прокси шумит именно там, где проблема.
   → Внедрён с МАЛЫМ весом (≤0.15, tie-breaker), не основной сигнал.

Вывод: reranker не лечит произношение (выбирает меньшее зло) → S16 (фонетический буткемп)
остаётся топ-приоритетом; калибровка весов продолжится на следующем human-наборе.