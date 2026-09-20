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
2. ~~НОВЫЙ объективный прокси: акустическая дистанция~~ — **ОПРОВЕРГНУТО контролем (20.09)**:
   `acoustic_emo_dist.py` (f0_med/f0_range/energy_dyn/voiced_share против teacher-профиля)
   на held-out teacher-клипах дал acc 0.18 ≈ шанс (1/7=0.14) — метрика НЕ работает даже
   в родном домене учителя (arousal-признаки не разделяют классы dialogs_emotional).
   Вывод: объективного эмо-гейта у нас нет и из дешёвых средств не появляется;
   единственные валидные инструменты — Gemini-судья + human A/B (S13).
   Артефакты: emo_teacher_profile.json, local_tests/acoustic_ctrl_* (контроль).
3. pairwise A/B против v1.0 (S13 listen-формат) — человек подтверждает.
4. Регресс: GATES §1 полный прогон (TTS/clone/DSP не должны упасть).

## Whisper-фикс (отдельно) — ПРОВЕРЕНО 20.09
Гипотеза «громкие teacher-шёпоты» **НЕ подтвердилась**: whisper_audit.py (по 60 клипов
на эмоцию) — whisper тише медианы прочих эмоций на **7.5 дБ** (WHISPER_AUDIT.md).
Следствие: крик вместо шёпота — дефект следования ИНСТРУКЦИИ моделью (speaking_mode
не отделена от эмоции в обучении), а не грязь данных. Фикс S12 = ось speaking_mode
с явными контрастными парами (normal vs whisper тот же текст/спикер) + фильтр НЕ нужен.

## Gate S12
- эмо-годен ≥ 55% (было 48.3%), sad ≥ 7/12 (было 4/12) — Gemini-судья, эмо-протокол 60;
- whisper: судья «шёпот похож на шёпот» ≥ 8/10 проб (сейчас 0 — крик);
- intensity-монотонность: ПАРНОЕ сравнение (тот же текст+ref, seed): gen(intense) vs gen(subdued) —
  RMS/F0-range выше у intense в ≥70% пар. Это прямое измерение, НЕ акустическая дистанция
  (которая опровергнута контролем);
- GATES §1 не хуже канона.

## Артефакты (ожидаемые)
- `s12_mix_build.py`, `v12_emo_mix/`, `whisper_audit.py`, `acoustic_emo_dist.py`
  (опровергнут как гейт — оставлен как диагностический инструмент), S12_RESULTS.md.