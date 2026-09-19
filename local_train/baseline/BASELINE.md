# BASELINE — воспроизводимая исходная версия (шаг 1 плана s2)

Дата фиксации: 15.09.2026. Снимок: `local_train\baseline\` (configs, data_snapshot, src_snapshot,
local_train_scripts, requirements-freeze.txt, CHECKSUMS.txt).

## Цепочка происхождения весов (КРИТИЧНО для слияний и resume)

```
AuK original (апстрим)
  └─ ckpts\AuK\auk_base.safetensors             sha256 29C65C0C…B8614  (не изменялся)
       └─ s0: LoRA r16, translit-данные, run_ru\
            └─ run_ru\auk_ru_best.safetensors   sha256 BB637011…E2AA6
                 ├─ s1: LoRA r32/α64, init = ЭТОТ файл, run_ru_s1\ (20569 клипов, 2 эпохи, seed 7)
                 │    └─ model_{9000,9500,10000,14750,15000,15250,15500,18000,22000}.pt
                 │       + model_last.pt (update 22000); адаптеры содержат peft + extra_state_dict (fusion)
                 └─ merge базис для ВСЕХ s1-мержей:
                      merged\auk_ru_{10000,18000,22000}.safetensors
```

### Правила (обязательные)
1. **Мерж s1-адаптера выполнять ТОЛЬКО с `--base_ckpt run_ru\auk_ru_best.safetensors`.**
   `merge_lora.py` по умолчанию берёт `ckpts\AuK\auk_base.safetensors` → молча даст ДРУГУЮ модель.
2. **Resume обучения** обязан передавать тот же `--init_ckpt` (wrapper
   `run_train_s1_resilient.ps1` подставляет `run_ru\auk_ru_best.safetensors`; train.py:629
   перезагружает базу из init_ckpt при каждом старте). Env-переменная `AU_RU_INIT` не
   использовалась (не задавалась в обёртках).
3. **Адаптер = peft-веса + fusion-параметры** (`extra_state_dict`: txt_proj, audio_embed,
   time_embed, norm_out, proj_out, layer_weights, layer_scale). При ручной сборке LoRA
   сохранять оба словаря (train.py:257, merge_lora.py:86).
4. Старый адаптер никогда не накладывать повторно поверх уже слитых весов.
5. **S2-стадия: init = `run_ru_s1\merged\auk_ru_10000.safetensors`** (не s0!).
   Диагностика 16.09: `run_ru\auk_ru_best.safetensors` — ТРАНСЛИТ-модель (s0), на кириллице
   даёт мусор («смер что дерматьева значит друба» vs u10000 «сбер что для мальцева значит
   дружба»). Тех-прогон s2 с s0-init давал брак на всех шагах; с u10000-init — чистые сэмплы.
   Мерж s2-адаптера: `--base_ckpt run_ru_s1\merged\auk_ru_10000.safetensors` (тот же файл).

## Версии окружения (ключевые)
python 3.10 (venv G:\AI\AuK\.venv) · torch 2.7.1+cu128 · torchaudio 2.7.1+cu128 ·
transformers 4.57.6 · peft 0.20.0 · accelerate 1.15.0 · gradio 6.10.0 · onnxruntime 1.23.2 ·
huggingface_hub 0.36.2. Полный список — `baseline\requirements-freeze.txt`.

## Чекпоинты и контр. суммы
Все sha256 — в `baseline\CHECKSUMS.txt` (15 файлов: base, s0-best, 10 адаптеров, 3 мержа).
Хранимые адаптеры: 9000, 9500, 10000 (исторические), 14750/15000/15250/15500 (финалисты),
18000 (RELEASE CANDIDATE), 22000 (последний), model_last (22000).

## Где снимались метрики/решения
- `USER_NOTES.md`, `RESEARCH_LOG.md`, `MAJOR_CHECKPOINTS.md`, `GOAL.md`, `user_reviews\*`,
  `local_tests\{keeper,e2e_suite,voices_u18000,ab_final,regress_*}`.
- Текстовый обработчик: `pron_lexicon.json` + `_accentize_ru/_tts_wrap` (дефисный респеллинг —
  экспериментальная опция, см. шаг 7 плана s2).

## Данные
- s1: `data\train.jsonl` (8.0 MB) + `data\val.jsonl`; копии — `baseline\data_snapshot\`.
- Источники s1: русские корпуса из `G:\AI\kyutai-ru\data\` (паспорт — шаг 4 плана).
