# ROADMAP AuK-ru — после S7@5750 (утверждён пользователем 19.09.2026)

Принцип: закрываем конкретные bottleneck'и, а не «учим дольше». Каждая стадия имеет явный gate.
Разделение: **версия продукта** (AuK-ru v1.0, v1.1…) ≠ **номер экспериментальной стадии** (S7, S8…).
Пользователь модели знает версию; стадии — внутренняя кухня R&D.

## Текущая фиксация

**AuK-ru v1.0 = s7@5750 + inference recipe** (см. ниже). Полный GATES §1 закрыт:
TTS WER 0.077, first_ok 0.812, clone sim best-of-3 0.771, clone WER 0.079, DSP 6/6,
произношение (phonetic-пак, судья): accent 9.81 / palatalization 9.69 / stress 9.56,
эмо-протокол годен 48.3% (vs 36.7% s5).
Известная плата: neutral clone/phonetics −0.6 по судье vs s5@4500; на ТРУДНЫХ текстах
произношение заметно хуже (accent ~7.9–9.0, naturalness ~5.3) — см. поправку ниже и S16.

Inference recipe v1.0: best-of-3 seeds (7,123,999), reranking по composite score,
normalize_rms + limit_peak, max_ref_seconds=30.

> ⚠️ ПОПРАВКА (20.09, S13): гейт «китайский акцент 0/N» НЕВАЛИДЕН — судья ни разу не
> использовал эту формулировку (0 упоминаний за все прогоны), поэтому метрика была
> структурно зелёной. Реальная проблема произношения существует на ТРУДНЫХ текстах
> (hard-набор A/B): accent падает 9.8→7.9, palatalization 9.7→7.9, брак 24/78;
> фонемные подмены ы→и («свои→сои»), палатализация («встреча→стрича») — подтверждены
> судьёй и пользователем. Штатный фонетический пак (16 лёгких фраз) её маскирует.
> Подробности: PRONUNCIATION_AXES.md, S13_PHONETICS.md. Новый гейт произношения —
> «доля hard-файлов с accent≤5 и palatalization≤5» + human A/B (см. S16).

## S16 — Phonetic bootcamp (ТОП-ПРИОРИТЕТ; методология v2 по итогам S13, 20.09)

### Переформулировка проблемы
Не «акцент вообще», а **нестабильная артикуляция на фонетически и лексически трудном
материале**. Целевой объект обучения — **фонетический контраст/семейство**, а не конкретное
слово или строка: если ломается «холодильник», учим семейство (стечения согласных + мягкость
+ редукции), а не повторяем это слово тысячи раз.

### Заморозка измерительного набора (ПРОВЕРЕНО)
`hard_cases_ru.jsonl` (241 текст) стал измерительным прибором → **заморожен как hard_eval_v1**:
- `local_train/hard_eval_v1/hard_eval_v1.jsonl` (241, sha256 в FROZEN.md),
- `hard_eval_v1_clean.jsonl` (**235** — 6 позиций исключены: уже встречаются в v7-базе,
  иначе гейт мерил бы запоминание, а не произношение),
- обучение на этих текстах ЗАПРЕЩЕНО; проверка перед каждым прогоном:
  `check_hard_eval_freeze.py` (exact + 4-граммы) и `freeze_hard_eval.py`.
- S16b-добавка проверена: 0 exact / 0 4-gram совпадений с frozen.

### Шесть корзин буткемпа (train — ДРУГИЕ предложения и формы)
| корзина | что тренируем | пример контраста |
|---|---|---|
| yi | ы↔и минимальные пары | был/бил, мыл/мил, рык/рик, мыши/миши |
| softness | твёрдость/мягкость, палатализация | ключи ≠ клучи, чн/чт/щн, н-ь перед и/е |
| clusters | стечения согласных | ств/нств/рств/нкт/ртр/здн/мпл |
| devoicing | оглушение и редукция с потерей фонем | хлеб/снег/дуб; ложка/шапка; вторая ≠ фарая |
| stress | ударения и омографы | з+амок/зам+ок, атл+ас/+атлас, начал/начал+а |
| long_num | длинные/составные слова + числа/даты | рентгеноэлектрокардиографический; 21.09.2026 |
+ отдельная корзина topwords — реальные top-error слова (сегодня, университета, тебя,
холодильник, кажется, сколько, работает…) **но преимущественно через склонения, производные
и новые контексты**, а не повтором одной строки.

### Метрики S16 (новых раньше не было)
- `human_mumble_rate` — доля файлов с человеческим флагом «жуёт слова»;
- `both_bad_rate` — доля пар слепого A/B с вердиктом «оба плохи»;
- ошибки ы/и; ошибки hard/soft; пропуски фонем/слогов; неправильные ударения;
  repetitions/stumbles; hard-text text_fidelity.

### Роль Gemini-судьи (изменена, ПРОВЕРЕНО согласованностью 21%)
Судья — **НЕ финальный гейт произношения**. Пайплайн вердикта:
```
automatic screening (ASR + оси судьи + mumble-прокси)
   → candidate shortlist (подозрительные + случайная выборка)
   → human blind test (listen_s16.html, флаги «жуёт» + A/B/оба плохи)
   → PASS / FAIL
```
Инструменты: `s16_human_gate.py` (shortlist + слепой HTML + SECRET-карта),
`s13_pairwise_score.py` (подсчёт с раскрытием).

### Gate S16 (главная цель — не «понизить WER»)
> Снизить human-detected mumbling и фонемные подмены на frozen hard set
> БЕЗ ухудшения штатного TTS и возможностей S7.
- human: mumble_rate < 15% (v1.0: ~30%), both_bad_rate < 20% (v1.0: 45%);
- автоматика (triage): фонемных подмен на hard-наборе меньше baseline, mangled меньше;
- не-регресс: TTS WER ≤ 0.077, first_ok ≥ 0.812, эмо годен ≥ 48.3%;
- **жёсткий rollback**: если hard-произношение растёт, но штатный TTS падает — S16 НЕ канон,
  а новая ветка роутинга: `normal TTS → S7 | clone → S5 | emotion → S7 | hard RU text → S16`.

### Артефакты
`freeze_hard_eval.py`, `check_hard_eval_freeze.py`, `hard_eval_v1/` (FROZEN.md + sha256),
`phonetic_baskets.py`, `s16_deficit_scan.py`, `s16_verify_deficit.py` (ASR-верификация
кандидатов: GigaAM WER ≤0.10 и отсутствие подмен ы→и/ч-ц/ш-щ), `s16b_mix_build.py`,
`s16_human_gate.py`, `watch_s16.ps1`, `s16_gate.ps1`, `aggregate_s16.py`, S16_RESULTS.md.

## Этапы (порядок утверждён)

СТАТУСЫ на 20.09 23:30: S8 ЗАКРЫТ (2 итерации, гейт не закрыт — S8B_RESULTS.md);
S9 ЗАКРЫТ (frontend + benchmark 241 + пробы, wer_mean 0.319→0.255 — S9_FRONTEND.md);
S10 ЗАКРЫТ (RFT 750 шагов: гейт 4/5, first-shot цель ❌ 0.7362; 111 пар намайнены —
S10_RESULTS.md); S11 ЗАКРЫТ v1 (composite reranker + generate_v1.py; калибровка по human-флагам
S13 начата: text_fidelity r=−0.583, ASR-прокси слаб recall 0.50 — S13_CALIBRATION.md);
S12 ЗАКРЫТ (intensity заработал 0.73 ✅, whisper-крик устранён, но шёпота нет и TTS-дрейф;
гейт 1/6 — S12_RESULTS.md); **S13 ЗАКРЫТ** (human A/B 40 пар + произносительный прогон судьи
на 80 файлах — S13_HUMAN_AB.md, S13_PHONETICS.md, PRONUNCIATION_AXES.md);
S14 ИЗМЕРЕН (RTF tts 0.67@nfe32, draft 0.64@nfe16 — S14_RTF.md); S15 ЗАКРЫТ (parity 1.16e-10,
композиция 0/5 — интерференция; РЕШЕНИЕ: маршрутизация route_infer.py, demo 8/8 — S15_RESULTS.md);
**S16 В РАБОТЕ** (методология v2: frozen hard_eval_v1, 6 корзин контрастов, human-gate).

### Интерпретация S13 (важно, исправляет прежнюю)
- **s8b 12 : s7 10 НЕ является победой s8b.** Главный сигнал теста — **18/40 = 45% пар
  «оба плохи»**: в половине случаев выбирать победителя бессмысленно, оба варианта с браком.
  Это и есть класс дефекта, который прежние aggregate-гейты систематически пропускали.
- Флаги «жуёт слова»: s7 11/40 (28%), s8b 13/40 (32%) — дефект общий для ветви, не
  специфичен для чекпойнта. Согласованность человек↔судья 21% → Gemini не финальный гейт.
- «Китайский акцент» как индикатор удалён из сравнительных таблиц (S7/S10/ROADMAP/STATUS);
  исторические упоминания оставлены с примечанием о невалидности.

### Порядок приоритетов (обновлён 20.09)
**S16 phonetic bootcamp → human hard-gate → pronunciation-aware reranker calibration →
frontend v2 → first-shot reliability (S10-v2).**
First-shot сознательно опущен: нет смысла делать первый seed стабильнее, если он стабильно
может сказать «пять→пить» или «холодильник→халильник».

**ИТОГ ВЕТКИ ОБЩИХ ДООБУЧЕНИЙ (ПРОВЕРЕНО 4 экспериментами S8/S8b/S10/S12):
продолжение LoRA на СМЕСЯХ ОБЩЕГО НАЗНАЧЕНИЯ после s7@5750 системно ухудшает TTS first_ok
и эмо-годен. s7@5750 — локальный оптимум, канон v1.0. S16 — исключение по замыслу: узкая
фонетическая задача + жёсткий rollback в роутинг (`hard RU text → S16`), а не замена канона.**

### S8 — Neutral Recovery (ЗАКРЫТ: подход не сработал)
Цель: сохранить эмоции S7, вернуть neutral clone/phonetics ≥ S5@4500.
Итог: S8@7750 → 2/5 гейтов; S8b@8500 (исправленный микс) → 0/5. Продолжение обучения
после s7 даёт дрейф TTS/эмоций независимо от состава данных. Единственный устойчивый
выигрыш — first-seed similarity (0.7378→0.7524), учтён в мотивации S10.
Детали: S8_RESULTS.md, S8B_RESULTS.md.

### S9 — Russian Frontend (веса не трогаем)
Конвейер: raw text → normalizer → stress/pronunciation → AuK.
Покрытие: числа с падежами, даты, валюты, телефоны, URL, аббревиатуры (RTX 6000 → «эр-тэ-икс
шесть тысяч»), ФИО, ё/е, омографы, ударения, code-switching RU/EN, названия моделей.
Артефакт: `hard_cases_ru.jsonl` (300–500 злобных предложений) — постоянный regression benchmark.
Инструмент-заготовка: thirdparty/rutextnorm (уже в дереве, в инференс не подключён).

### S10 — Single-shot Reliability
Проблема (ПРОВЕРЕНО, seed-проба 25 клонов × 3 seeds на s7@5750):
first-seed median 0.7378 vs best-of-3 0.7711; gap mean +0.037; в 8/25 первый seed уже лучший;
sim<0.75: first 13 → best 9; sim<0.6: first 2 → best 0.
Цель: сократить разрыв «первая генерация − лучшая из трёх», метрика = first-shot success rate.
Метод: automatic candidate mining (4–8 кандидатов на prompt → ASR + sim + DSP + repetition/artifact
детекторы → BEST/MID/BAD) → preference dataset {prompt, reference, chosen, rejected} → обучение.

### S11 — Quality-aware Inference (без retraining)
Composite reranker вместо чистого WeSpeaker:
score = 0.35·sim + 0.25·ASR_fidelity + 0.15·pronunciation + 0.10·loudness + 0.10·pause_quality
+ 0.05·naturalness − repetition_penalty − artifact_penalty.
Веса калибруются на human-labelled наборе. Мотивация: max sim ≠ лучший перцептивный результат
(«голосом похож, но заговаривает слово»).

### S12 — Emotion v2 + speaking_mode
Разделить: emotion (neutral/happy/sad/angry/…), intensity (continuous 0..1), style
(restrained/controlled/…). Whisper/breathy/shouting — НЕ эмоции, а отдельная ось speaking_mode
(наблюдение: whisper-инструкция давала крик — архитектурно чище вынести).
Данные: sad — худшая ось s7 (4/12 годен); continuous intensity вместо бинарности.

### S13 — Human Preference Benchmark
Постоянный набор: 50 текстов × 10 спикеров × 2–3 версии модели. Только pairwise A/B
(естественность / дикция / похожесть на reference / эмоция / «стал бы использовать»).
Никаких «оценок 1–10». Gemini — массовая фильтрация, не ultimate metric;
human listening — обязательный gate (подтверждено историей s5: aggregate-метрики не ловили
«заговаривания», громкость между seeds, whisper-крик).

### S14 — Real-time / Fast Inference
bf16/fp16, quantization, torch.compile, FlashAttention, batching, KV-cache, CUDA graphs,
AuK-Flash, SGLang/serving, streaming. Цель: RTF < 1, затем 0.3–0.5.

### S15 — Adapter Architecture
Перестать выпускать монолиты 6.1 ГБ на каждый эксперимент. AuK-ru Core + переключаемые
адаптеры: Neutral / Emotion / Whisper / Character / Narration / Experimental
(`model.load_adapter("emotion")`).

### Долгосрочно — AuK-ru 2
Продуктовая архитектура: Russian frontend → AuK-ru Core → {Voice cloning, Emotion adapter,
Style adapter} → quality-aware generation (N кандидатов) → reranker → WAV.

## Aniemore/SER — закрытая ветка
Emotion-гейт на синтетике невалиден (disgust-collapse, доменный мисматч; RESD-контроль 99.3%).
Дальнейшие SER-модели НЕ оптимизируем; вместо них — human-calibrated preference/eval (S13).

## Ключевой принцип этапа
Следующий этап — не «больше обучения», а «меньше случайности и меньше специальных случаев,
которые ломают хороший в целом голос».