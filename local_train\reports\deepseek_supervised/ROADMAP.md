# ROADMAP AuK-ru — после S7@5750 (утверждён пользователем 19.09.2026)

Принцип: закрываем конкретные bottleneck'и, а не «учим дольше». Каждая стадия имеет явный gate.
Разделение: **версия продукта** (AuK-ru v1.0, v1.1…) ≠ **номер экспериментальной стадии** (S7, S8…).
Пользователь модели знает версию; стадии — внутренняя кухня R&D.

## Текущая фиксация

**AuK-ru v1.0 = s7@5750 + inference recipe** (см. ниже). Полный GATES §1 закрыт:
TTS WER 0.077, first_ok 0.812, clone sim best-of-3 0.771, clone WER 0.079, DSP 6/6,
«китайский акцент» 0/100, эмо-протокол годен 48.3% (vs 36.7% s5).
Известная плата: neutral clone/phonetics −0.6 по судье vs s5@4500.

Inference recipe v1.0: best-of-3 seeds (7,123,999), reranking по composite score,
normalize_rms + limit_peak, max_ref_seconds=30.

## Этапы (порядок утверждён)

СТАТУСЫ на 20.09 19:00: S8 ЗАКРЫТ (2 итерации, гейт не закрыт — S8B_RESULTS.md);
S9 ЗАКРЫТ (frontend + benchmark 241 + пробы, wer_mean 0.319→0.255 — S9_FRONTEND.md);
S10 ЗАКРЫТ (RFT 750 шагов: гейт 4/5, first-shot цель ❌ 0.7362; 111 пар намайнены —
S10_RESULTS.md); S11 ЗАКРЫТ v1 (composite reranker + generate_v1.py, калибровка весов
ждёт S13 — S11_RERANKER.md); S12 ЗАКРЫТ (intensity заработал 0.73 ✅, whisper-крик устранён,
но шёпота нет и TTS-дрейф; гейт 1/6 — S12_RESULTS.md); S13 ГОТОВ К ЗАПУСКУ (listen.html
40 пар + scorer; ждёт human-прослушивания); S14 ИЗМЕРЕН (RTF tts 0.67@nfe32 <1 ✅,
оптимизация до 0.3–0.5 — будущая работа: compile/flash/batching); S15 ЗАКРЫТ (parity 1.16e-10,
композиция 0/5 — интерференция; РЕШЕНИЕ: маршрутизация route_infer.py, demo 8/8 —
S15_RESULTS.md).

**ГЛАВНЫЙ ИТОГ ВЕТКИ ДООБУЧЕНИЙ (ПРОВЕРЕНО 4 экспериментами S8/S8b/S10/S12):
любое продолжение LoRA после s7@5750 системно ухудшает TTS first_ok и эмо-годен.
s7@5750 — локальный оптимум; канон v1.0 окончательный. Дальнейший прогресс:
inference-уровень (S11/S15 — работают), human-калибровка (S13), новые данные
(контрастные whisper-пары), оптимизация скорости (S14).**

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