# Prospective quality gates и предлагаемое вмешательство (16.09.2026)

Статус документа: **prospective** — сформулирован ПОСЛЕ измерений, не выдаётся за
предрегистрированный ранее. Прошлые допуски (`AB_PROTOCOL.md`) остаются как paired-часть.

## 1. Абсолютные пороги (в дополнение к paired nonregression)

Высокое относительное качество не спасает, если абсолют хуже порога. Пороговые значения —
предложение к утверждению root'ом; применяются на held-out, не на dev.

| Область | Метрика | Абс. порог | Текущий факт |
|---|---|---|---|
| TTS (RU) | WER абс. | ≤ 0.20 | u10000 1.04; A@500 0.40; B@500 0.19 |
| TTS (RU) | вставки | ≤ 0.05 | требует пересчёта ASR с insertions |
| TTS (RU) | first_ok | ≥ 0.70 | A@250/B@500 0.667 |
| Клон | sim к реф. (med) | ≥ 0.75 | u10000 0.777; B@500 0.726 (ниже базовой!) |
| Клон | WER текста | ≤ 0.15 | B@500 0.422 (не проходит) |
| Клон | утечка реф. | ≤ 0.02 | v2-метрика: B@500 0.072, A@500 0 |
| Фонетика | замены ж/з, ч/ц, ы/и | ≤ 0.10 на верифицированных парах | **не измерено надёжно** |
| Инструмент volume | \|delta_db − target\| | ≤ 2.0 дБ, знак верный, clip ≤ 0.001 | u10000 +1.37; B@500 +1.99 (при +6) → FAIL |
| Инструмент speed | \|rate − target\|/target | ≤ 0.10 | B@500 up 1.150, down 0.893 → PASS |
| Инструмент pitch | \|semitones − target\| | ≤ 1.0 | ±3.7…4.3 при ±2 → FAIL |
| Инструмент denoise | snr_gain | ≥ +3.0 дБ и corr ≥ 0.8 | все варианты **отрицательны** (−3.6…−5.8) → FAIL |

Правило: **нельзя** ставить «проход», если WER ≥ 1.06 (как у u10000) или если инструмент
объявлен «лучше A», при этом оба не выполняют абсолютную операцию.

## 2. Paired nonregression (сохраняется)
- Речь: ухудшение WER ≤ +0.02 абс. против базы; без новых клиппингов.
- Клон: sim не ниже базы на том же Dev; leak не растёт.
- Инструменты: B не хуже A; и абсолютный порог §1 обязателен.

## 3. Held-out
Финальный held-out — speaker-disjoint (новые кластеры, не пересекающиеся с s2 train).
Текущий tools val НЕ held-out: 2242 val-строки имеют corpus-split=train, 317 кластеров
пересекаются с s2 train (`TOOLS_OVERLAP.md`). До пересборки val абсолютные пороги по
инструментам не могут быть итоговыми.

## 4. Предлагаемое вмешательство (маленькое, воспроизводимое)

Диагноз по замерам: s2-B улучшил речь и speed, но не выучил volume/pitch/denoise; цель — не
новый аудит, а короткий дообучающий прогон от лучшего чека.

- **База/адаптер:** продолжать LoRA от `run_s2_B/model_500.pt` (init = `run_ru_s1/merged/auk_ru_10000.safetensors`),
  r32/α64, тот же train.py; НЕ полный fine-tune.
- **LR:** 1e-5 (ниже s2, чтобы не сломать речь), warmup 25.
- **Данные:** та же речь (`v2_after_identity`), но tools-долю поднять до ~30% и
  перебалансировать по операциям (volume и pitch сейчас недопредставлены в эффекте):
  volume_up/down по 3200, pitch_up/down по 3200, speed по 1600, denoise — исключить
  до пересборки цели (текущая цель `noise_add` «Remove the background noise» даёт
  отрицательный snr_gain даже у обученных вариантов).
- **val:** пересобрать speaker-disjoint (исключить 317 общих кластеров), 400-800 строк.
- **Бюджет:** 2000 обновлений; save 250/500/1000/2000. По факту s2: 500 обновлений ≈ 8 мин
  GPU при ~0.9 ч аудио; 2000 обновлений ≈ 30-40 мин GPU и ~3.5 ч аудио.
- **Стоп-условия:** ранняя остановка, если на dev WER речи +>0.02 абс. или speaker sim <0.75;
  иначе фиксированные 2000.
- **Хранилище:** +6.1 ГБ на каждый merged (4 промежуточных ≈ 25 ГБ); адаптеры 0.6 ГБ.

### Что сделать ДО прогона (этой фазы не выполняется)
1. Пересобрать tools val speaker-disjoint и проверить content-hash (скрипт `tools_overlap.py`).
2. Перегенерировать цели volume/pitch с точным DSP-контролем (volume строго ±6 дБ, pitch ±2 st),
   denoise — цель от реального денойзера; проверить `tool_dsp_measure.py` на обучающих парах.
3. Прогнать upstream-контроль на валидных английских фикстурах (`original_eval_manifest.json`),
   чтобы отделить «адаптация сломала арсенал» от «модель не умела».

### Точная следующая runnable-команда (после пересборки данных)
```powershell
.venv\Scripts\accelerate.exe launch --num_processes 1 --num_machines 1 --mixed_precision bf16 `
  -m auk.train.train --train_jsonl local_train\data_s2_full\v3_fix\B_mix_70_30\train.jsonl `
  --val_jsonl local_train\data_s2_full\v3_fix\val_speaker_disjoint.jsonl `
  --config local_train\s2_config.yaml `
  --init_ckpt local_train\run_ru_s1\merged\auk_ru_10000.safetensors `
  --resume_adapter local_train\run_s2_B\model_500.pt `
  --output_dir local_train\run_s3_B2 --learning_rate 1e-5 --max_updates 2000 --warmup_steps 25 `
  --frames_threshold 384 --max_samples 2 --save_per_updates 250 --logging_steps 10 --val_per_updates 250 `
  --seed 7 --lora True --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --use_ema False --bf16_transformer True
```
(`--resume_adapter` — предлагаемый флаг; при отсутствии поддержки resume делать через init_ckpt
уже слитого чекпоинта. До реализации — уточнить в train.py.)

## 5. Оценка после прогона
- `tool_dsp_measure.py` (абсолютные пороги §1) на новой speaker-disjoint выборке.
- ASR WER с вставками на held-out речи.
- Автосудья v3 (strict schema) как вспомогательный сигнал; слепое прослушивание опционально.
- Никакие числа этой фазы не считать решением: полный автосудья — сигнал, не приговор.
