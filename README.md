# AuK — русская LoRA-адаптация (Ru)

Форк [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK) (MIT) —
инструктивная TTS/аудио-модель (Flux2Edit + Qwen2.5-Omni-3B + BigVGANFlowVAE) —
с русской адаптацией через цепочку LoRA-обучения.

> ВНИМАНИЕ: это **форк с локальными изменениями** для русского языка и эмоциональной
> выразительности. Веса и датасеты **не входят** в этот репозиторий; обученные модели
> опубликованы на Hugging Face: [aga7on/AuK-ru](https://huggingface.co/aga7on/AuK-ru).
> Аудио-референсы и датасеты остаются приватными.
>
> Оригинальный README апстрима: https://github.com/Tencent-Hunyuan/AuK

## Семплы — эмоции (s7@5750) — слушать

Пять генераций с эмоциональной инструкцией («say in Russian with a {emotion} tone»),
референсы — открытые корпусные голоса (не авторов репозитория). Все пять — verdict «годен»,
overall 9/10 у автосудьи (Gemini). **Кликните по ссылке — откроется страница файла со встроенным
аудиоплеером GitHub** (скачивать не нужно):

| ▶ слушать | голос | эмоция |
|---|---|---|
| [▶ 01_female_happy.wav](samples/01_female_happy.wav) | женский | радость |
| [▶ 02_female_excited.wav](samples/02_female_excited.wav) | женский | восторг |
| [▶ 03_male_sad.wav](samples/03_male_sad.wav) | мужской | грусть |
| [▶ 04_male_angry.wav](samples/04_male_angry.wav) | мужской | злость |
| [▶ 05_male_fearful.wav](samples/05_male_fearful.wav) | мужской | страх |


## GATES §1 — s7@5750 (эмоциональный этап, полный прогон 19.09.2026)

| метрика | s5@4500 (канон) | s7@5750 | гейт |
|---|---|---|---|
| TTS WER (GigaAM, mean) | 0.081 | **0.077** | ≤0.20 ✅ |
| TTS first_ok | 0.750 | **0.812** | ≥0.70 ✅ |
| clone sim (median, 1-seed) | 0.742 | 0.741 | — |
| clone sim (best-of-3) | 0.763 | **0.771** | ≥0.75 ✅ |
| clone WER (mean) | 0.055 | 0.079 | ≤0.15 ✅ |
| DSP (volume/speed/pitch) | 6/6 PASS | 6/6 PASS | ✅ |
| «китайский акцент» (clone100) | 0/100 | 0/100 | 0 ✅ |
| Судья: эмо-протокол годен | 36.7% | **48.3%** | — |
| Судья overall: tts / clone / phonetics | 6.90 / 8.20 / 8.06 | 7.06 / 7.56 / 7.44 | tts +0.16, clone −0.64, phonetics −0.62 |

Эмо-гейт Aniemore на синтетике невалиден (disgust-коллапс, доменный мисматч;
RESD-контроль 99.3%) — эмоции подтверждены автосудьёй и прослушиванием.

## 🏷️ AuK-ru v1.0

**Канонический релиз: s7@5750** (`auk_s7_5750.safetensors` на HF) — полный GATES §1 закрыт
(TTS WER 0.077, first_ok 0.812, clone sim best-of-3 0.771, clone WER 0.079, DSP 6/6,
«китайский акцент» 0/100, эмоции годен 48.3%). Рецепт инференса: best-of-3 seeds (7, 123, 999)
+ reranking по speaker similarity + normalize_rms + limit_peak. Fallback нейтрального
clone/phonetics — s5@4500. План развития: [ROADMAP.md](ROADMAP.md).

## Что внутри

- `src/` — код модели + инференс (изменения и дополнения к апстриму):
  - `infer_auk.py` — инференс: авто-обрезка длинных референсов (`max_ref_seconds=30`),
    нормализация громкости (`normalize_rms`, RMS −20 dBFS);
  - `quality.py` — объективные замеры (WER через GigaAM-ASR, DSP-метрики);
  - `ru_translit.py` — транслитерация (этап s0);
  - `gigaam_ctc.py` — локальный ASR-контроль (GigaAM).
- `local_train/` — конфиги, скрипты слияния/обучения/оценки, отчёты раундов
  (без чекпойнтов, датасетов и секретов).
- `LICENSE` — оригинальная лицензия Tencent (MIT), сохранена как требуется.

## Russian frontend (S9)

Письменный русский → произносимый текст перед моделью:

\\python
from auk.infer.ru_frontend import to_speakable
to_speakable("Встреча 21.09.2026 в 14:30, тел. 8-800-555-35-35")
# -> "Встреча дв+адцать п+ервое сентябр+я ... телеф+он в+осемь восемьс+от ..."
\
Конвейер: rutextnorm (числа/даты/валюты/единицы) → tech-prespell (URL/email/версии/аббревиатуры)
→ safe_accentize (ударения RUAccent с защитой от фонетического респеллинга) → опц. транслит.
Regression-бенчмарк: \local_train/hard_cases/hard_cases_ru.jsonl\ (241 текст, 10 категорий),
прогон — \local_train/frontend_probe.py\. Результаты: S9_FRONTEND.md (wer_mean 0.319→0.255).

## Быстрый старт (v1.0)

Единая точка входа с полным рецептом (frontend → best-of-3 → composite rerank → нормализация):

\\powershell
# TTS (нормализация чисел/дат/телефонов встроенная)
python local_train\generate_v1.py --text "Встреча 21.09.2026 в 14:30." --out out_tts

# клонирование голоса + эмоция
python local_train\generate_v1.py --text "Привет, как дела?" --ref ref.wav --emotion happy --out out_clone
\
Режимы: TTS (без --ref), clone (--ref), clone+emotion (--emotion happy|sad|angry|fearful|excited|...).
Best-of-N seeds (7,123,999) с composite-скорингом; лучший копируется в \1_best.wav\.
Веса: HF \ga7on/AuK-ru\ → \uk_s7_5750.safetensors\ + \config.yaml\ в \local_train/run_s7/merged/\.

## Статус R&D (20.09.2026)

Дорожная карта S8–S15 исполнена; все отчёты в local_train/reports/deepseek_supervised/ (ROADMAP.md — сводка):

| этап | итог |
|---|---|
| S8/S8b neutral recovery | гейты не закрыты (2/5, 0/5) — ветка закрыта |
| S9 Russian frontend | ✅ wer_mean hard-бенча 0.319→0.255, 241 текст, интеграция в инференс |
| S10 single-shot / preference | 111 пар намайнено; RFT 4/5, first-shot не достигнут — закрыт |
| S11 quality-aware inference | ✅ composite reranker + generate_v1.py (калибровка весов ждёт S13) |
| S12 Emotion v2 | intensity-ось заработала (0.73), whisper-крик устранён; гейт 1/6 — закрыт |
| S13 human A/B | 40 слепых пар готовы (listen.html), скорер опубликован — ждёт прослушивания |
| S14 RTF | ✅ TTS 0.67@nfe32 (<1); цель 0.3–0.5 — будущие оптимизации |
| S15 adapters | композиция весов 0/5 (интерференция); решение — маршрутизация (route_infer.py) |

**Мета-вывод (4 независимых эксперимента):** продолжение LoRA-обучения после s7@5750 системно
ухудшает TTS first_ok и эмоции — **s7@5750 = окончательный канон v1.0**; прогресс далее только
на inference-уровне (маршрутизация/reranker) и через human-калибровку.

## Линейка обучения (доказательная цепочка)

Каждый этап мержится ТОЛЬКО со своей базой (см. `LINEAGE` в отчётах):

| этап | адаптер → база | r/α | описание |
|---|---|---|---|
| s0 | — → `auk_base.safetensors` | 16/32 | транслитерация |
| s1 | `run_ru_s1/model_*.pt` → база s0 | 32/64 | русская речь |
| s2 (A/B) | `run_s2_*/model_*.pt` → база s1 | 32/64 | пилот, выбор ветки |
| s3 | → база s2-B | 32/64 | чистовая речь/клонирование |
| s4 | → база s3 | 32/64 | инструктивный набор |
| s5 @4500 | → база s4 | 32/64 | каноническая (safety-точка) |
| s7 | → база s1-10000 | 32/64 | эмоции (микс v7, emo-доля 0.27) |

Пример слияния:

```powershell
.venv\Scripts\python.exe local_train\merge_lora.py --run_dir local_train\run_s7 `
  --ckpt local_train\run_s7\model_6750.pt `
  --out local_train\run_s7\merged\auk_s7_6750.safetensors `
  --base_ckpt local_train\run_ru_s1\merged\auk_ru_10000.safetensors --lora_r 32 --lora_alpha 64
```

## Обучение

```powershell
& .venv\Scripts\accelerate.exe launch --num_processes 1 --num_machines 1 --mixed_precision bf16 `
  -m auk.train.train --train_jsonl <train.jsonl> --val_jsonl <val.jsonl> `
  --config <base>\merged\config.yaml --init_ckpt <base>.safetensors `
  --output_dir <run> --learning_rate 5e-6 --max_updates 6750 --warmup_steps 25 `
  --frames_threshold 384 --max_samples 2 --save_per_updates 250 --logging_steps 10 `
  --val_per_updates 250 --seed 7 --lora True --lora_r 32 --lora_alpha 64 `
  --lora_dropout 0.05 --use_ema False --bf16_transformer True
```

Resume: скопировать последний хороший `model_N.pt` в `model_last.pt`, перезапустить ту же команду
(train.py сам рескейлит lr оптимизатора к CLI-значению — важно, см. отчёты).

## Оценка качества

- Автосудья — локальный Gemini-прокси (ключ через env `AUK_GEMINI_KEY`, не коммитится).
- ASR-контроль — GigaAM (локально).
- Эмоциональный гейт — Aniemore WavLM; контроль на RESD: **99.3%** (150/150);
  на синтетических langswap-учителях гейт 17% → гейт верен только для RESD-подобных эмоций.
- Результаты раундов: `local_train/reports/deepseek_supervised/` (STATUS.md, S5_RESULTS.md и др.).

## Модели

| имя | файл | sha256 |
|---|---|---|
| канон s5@4500 | `auk_s5_4500.safetensors` | в model card на HF |
| s7@5000 / 5750 / 6750 | `auk_s7_*.safetensors` | в model card на HF |

Подробности загрузки и использования — [Hugging Face model card](https://huggingface.co/aga7on/AuK-ru).

## Поддержать проект 💸

Любая денежная помощь приветствуется — проект обучается на личном железе,
а автор мечтает о RTX 6000 (эмоции сами себя дальше не дообучат).

**USDT (TRC-20):** `TYXmYihm39f7Sboj3FCn5WXBPy89Tf6dYa`

**Обратная связь:** Telegram [@aga7onia](https://t.me/aga7onia) — вопросы, баги, пожелания и отчёты о прослушанных семплах.

Спасибо! Каждая монетка — это ещё несколько часов обучения и пара новых эмоций в модели.

## Конфиденциальность

- Голосовые датасеты и референсы пользователя в публичные репозитории **не включаются**.
- API-ключи — только через переменные окружения; в исходниках ключей нет.
- `SECRET_map.csv` (слепые тесты) никогда не загружается.