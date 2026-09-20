# S12 DESIGN — Emotion v2 + speaking_mode (проект, до запуска)

## Мотивация (данные из наших прогонов)
- s7 эмоции: годен 48.3% — лучший результат, но sad 4/12 — худшая ось; excited→enthusiasm
  путается с happy у судьи; «whisper превращался в крик» (пользовательское прослушивание s5).
- Aniemore-гейт невалиден на синтетике (disgust-collapse) → объективной метрики эмоций у нас
  НЕТ; судья Gemini измеряет «выражена ли эмоция» косвенно (verdict+issues).
- Учитель (langswap dialogs_emotional) содержит: happy/sad/angry/fear/disgust/surprise/
  whisper/laughing/neutral — т.е. РЕЖИМЫ (whisper, laughing) смешаны с эмоциями в одной оси.

## Ключевое решение: три независимые оси
```
emotion:      neutral | happy | sad | angry | fearful | disgusted | surprised
intensity:    0.2 | 0.5 | 0.8   (continuous, дискретизируется в 3 уровня для обучения)
speaking_mode: normal | whisper | breathy | laughing | shouting
style (позже): restrained | neutral | expressive
```
Инструкция-формат (совместим с teacher-форматом s7):
  «Reproduce the reference voice and say in Russian with a sad tone, gently (intensity 0.3), whispering: '<текст>'»

## Данные
1. Переразметка dialogs_emotional: whisper/laughing → speaking_mode, остальные → emotion.
   neutral_emo → intensity 0.2 (сдержанно).
2. Интенсивность: ПРОВЕРЕНО на подмножестве — у teacher-клипов есть громкостная/темп-вариативность;
   прокси-разметка intensity по RMS+F0-range+dur (без ручной разметки): low/mid/high квартили
   внутри (speaker, emotion). Валидация прокси: 30 клипов глазами/ушами + согласованность ≥80%.
3. Расширение: RESD (99.3% гейт-валидность) — 7 эмоций, можно как eval-домен для гейта.
4. Объём: v7-эмо 13824 строки → v12-эмо ~20000 (с intensity/mode вариантами инструкций,
   те же аудио — мультиинструктивное обучение).

## Обучение
- S12 стартует от ЛУЧШЕГО на момент запуска канона (v1.0/v1.1), НЕ от s8-ветки.
- 1500–2000 шагов lr 3e-6, save/val 250, гейт-прогон на 3 чекпойнтах (early/mid/late) —
  урок s7: оптимум эмоций был в середине (5750), финал (6750) регрессировал.

## Оценка (без Aniemore)
1. Gemini-судья эмо-протокол 60 (сравнение годен% против канона).
2. НОВЫЙ объективный прокси: F0-range + energy-динамика + speech-rate против teacher-статистик
   по (emotion) — «эмоциональная акустическая дистанция» (не классификатор, а сопоставление
   статистик; устойчиво к доменному сдвигу в отличие от Aniemore).
3. pairwise A/B против v1.0 (S13 listen-формат) — человек подтверждает.
4. Регресс: GATES §1 полный прогон (TTS/clone/DSP не должны упасть).

## Whisper-фикс (отдельно) — ПРОВЕРЕНО 20.09
Гипотеза «громкие teacher-шёпоты» **НЕ подтвердилась**: whisper_audit.py (по 60 клипов
на эмоцию) — whisper тише медианы прочих эмоций на **7.5 дБ** (WHISPER_AUDIT.md).
Следствие: крик вместо шёпота — дефект следования ИНСТРУКЦИИ моделью (speaking_mode
не отделена от эмоции в обучении), а не грязь данных. Фикс S12 = ось speaking_mode
с явными контрастными парами (normal vs whisper тот же текст/спикер) + фильтр НЕ нужен.

## Gate S12
- эмо-годен ≥ 55% (было 48.3%), sad ≥ 7/12 (было 4/12);
- whisper: судья «шёпот похож на шёпот» ≥ 8/10 проб (сейчас 0 — крик);
- intensity-монотонность: 0.8 > 0.5 > 0.2 по акустической дистанции в ≥70% пар;
- GATES §1 не хуже канона.

## Артефакты (ожидаемые)
- `s12_mix_build.py`, `v12_emo_mix/`, `acoustic_emo_dist.py` (прокси-метрика),
  `whisper_audit.py`, S12_RESULTS.md.