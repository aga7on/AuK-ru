# UPSTREAM WATCH — Tencent-Hunyuan/AuK (обновление 09.11.2026)

Собрано: `probe_upstream.py` (GitHub API) + страница проекта `auk-project.github.io`.
Проверено по API и файлам, не по пересказу. В репозитории ничего не менялось.

## Репозиторий (факты из API)

- default_branch: `main`, pushed_at: 2026-10-27, size: 22 089 KB, stars: 1 589
- поле license в API: **NOASSERTION** (GitHub не распознал лицензию автоматически;
  в файле `LICENSE` при этом текст MIT — см. ниже, это важно для нашего форка)

## Главное: да, инструмент для Flash вышел

**AuK-Flash = дистилляция в 2 стадии, и код лежит в `main` апстрима.**

| стадия | entry point | launch script |
|---|---|---|
| 1. consistency initialization | `src/auk/train/distill/initializer/train.py` | `scripts/train_init.sh` |
| 2. DMD (distribution matching distillation) | `src/auk/train/distill/dmd/train.py` | `scripts/train_dmd.sh` |

Общий код: `src/auk/train/distill/common.py`. Документация: **`docs/DISTILL.md`** (10 289 байт,
добавлена в `main`). Итог рецепта: студент сэмплит **4 шага без CFG**, заявленное ускорение
**4.5×** по wall-clock при сопоставимых условиях.

## Что известно точно (из DISTILL.md и train_dmd.sh)

| параметр | значение по умолчанию |
|---|---|
| `num_gpus` | **8** |
| `max_student_updates` | 2500 |
| `save_per_student_updates` | 500 (итог: `ckpts/auk_dmd/model_2500.pt`) |
| `fake_score_updates_per_student` | 5 (критик на каждый шаг студента) |
| `frames_threshold` / `max_samples` | 10000 / 24 |
| `teacher_cfg_scale` | 4.0 (в launcher) / 3.0 (дефолт конфига) |
| `regression_weight` | 1.0 |
| `--use_cd` | true (иначе re-noise ground truth) |
| `--task_routing` | false; при включении `--regression_only_tasks` = `ss,music,fisher` |
| `attn_backend` | `torch` (для `flash_attn` нужен подходящий wheel) |
| `--checkpoint_every_n_layers 1` | рекомендация для экономии VRAM |

**⚠️ Важно для нас: DMD тренирует ПОЛНУЮ модель, а не LoRA.** В `DMDConfig` нет полей
`lora*`; в `dmd/train.py` замораживаются только `vae_model` и `model.layer_weights` /
`model.layer_scale`, остальные параметры обучаются. То есть наша LoRA-схема (r32/α64 поверх
замороженной базы) с этим рецептом напрямую **не сочетается** — нужен полный файнтюн
студента, для которого дефолт 8 GPU.

Оценка применимости к нашей конфигурации (2×16 ГБ): **ГИПОТЕЗА, не проверено** — памяти
почти наверняка не хватит: студент полный (1.5B) + критик + учитель для teacher-walk.
Даже с `--checkpoint_every_n_layers 1` и `max_samples` = 1–2 это класс 40–80 ГБ.

## Ветки апстрима (проверено через API)

| ветка | последний коммит | дата | примечание |
|---|---|---|---|
| `main` | Merge PR #30 (docs/shared-model-card) | 2026-10-27 | **содержит весь distill-код** |
| `dev-wenming` | docs(readme): remove duplicate Prompt Enhancer entry | 2026-09-11 | только документация |
| `dev-ziyang` | announce vLLM-Omni and audio.cpp support | 2026-09-25 | анонс поддержки vLLM-Omni / audio.cpp |
| `feat/PE` | feat: add prompt enhancer CLI | 2026-09-07 | Prompt Enhancer (CLI) |
| `feat/local-pe` | feat(pe): run the Prompt Enhancer offline with `--local` | 2026-09-18 | PE без облака |
| `feat/mlx-apple-silicon` | feat(mlx): native MLX inference backend for Apple Silicon | 2026-09-13 | MLX только для инференса |

Ветки — вспомогательные: distill подтянут именно в `main`, отдельной «flash-ветки» нет.

## Что изменилось в `main` относительно нашей базы (0dfd4d3)

27 коммитов вперёд, 0 назад; затронуто 20 файлов. Существенное:

- **added** `docs/DISTILL.md`, `docs/SHARED.md`, `scripts/train_init.sh`, `scripts/train_dmd.sh`,
  весь `src/auk/train/distill/**` (~1330 строк);
- **modified** `src/auk/train/train.py`: в датасет добавлено поле `task`
  (`row.get("task")` и `"tasks"` в collate) — **наши правки это не ломает**: наш патч
  `val_max_rows` находится в `AukJsonlDataset.__init__`, другая часть файла;
- **modified** `src/auk/infer/infer_auk.py`: `model.to(float32)` → `model.transformer.to(float32)`
  (меньше памяти: VAE больше не поднимается в fp32);
- **modified** `src/auk/infer/pe.py`: уборка temp-файла на error-path;
- **modified** `scripts/train.sh`: убраны строки `unset http_proxy ...`.

Дифф мал и аддитивен — конфликтов с нашим форком нет.

## Расхождение с нашим пониманием архитектуры (стоит проверить)

Страница проекта описывает модель как **1.5B-параметровую**, с VAE **50 Гц** и
«hybrid rectified-flow Transformer: dual-stream MMDiT blocks → unified single-stream DiT»
(плюс post-training: human-feedback preference optimization и reward-based RL).
Наша рабочая база исторически описывалась как «Flux2Edit + Qwen2.5-Omni-3B + BigVGANFlowVAE».
**ГИПОТЕЗА:** это одна и та же модель в разных терминах (Flux2Edit — кодовое имя
MMDiT-блока; Qwen-омни — мультимодальный LLM-кондиционер; «50 Гц VAE» — наш BigVGANFlowVAE),
но я это не проверял сверкой кода. Если понадобится точность в документации —
это отдельная проверка, не требующая обучения.

## Что это значит для проекта

1. **Наш v1.0 (s7@5750) остаётся как есть.** Смысл Flash — ускорение инференса
   (4 шага вместо 64/32), а не качество.
2. **Обучить свой Flash на 2×16 ГБ нельзя:** нужен полный файнтюн студента + критик,
   дефолт 8 GPU; LoRA-версии этого рецепта апстрим не предоставил.
3. **Что можно забрать бесплатно:**
   - `--checkpoint_every_n_layers 1` как приём экономии VRAM (применимо к нашему `train.py`);
   - `model.transformer.to(float32)` вместо `model.to(float32)` — уменьшает память при
     инференсе; можно перенести в локальный `infer_auk.py` и **замерить** (проверяемая
     задача без обучения).
4. **vLLM-Omni / audio.cpp** (ветка `dev-ziyang`) — потенциально полезнее для нас, чем Flash:
   это про скорость и портируемость инференса, без требования 8 GPU на обучение.
5. **Для лицензионной части:** API отдаёт `license: NOASSERTION`. Значит нельзя
   утверждать «GitHub подтверждает MIT»; корректная формулировка — «в файле `LICENSE`
   апстрима текст MIT, но GitHub автоматически лицензию не распознал». Формулировки
   в наших README этому уже соответствуют (там ссылка на файл `LICENSE`, а не на API).

## Ссылки

- Страница проекта: https://auk-project.github.io/
- Технический отчёт (arXiv): https://arxiv.org/abs/2609.08936
- Апстрим: https://github.com/Tencent-Hunyuan/AuK (`main`, `docs/DISTILL.md`)