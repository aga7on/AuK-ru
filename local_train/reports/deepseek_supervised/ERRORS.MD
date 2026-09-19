# ERRORS.MD — журнал ошибок (AuK / supervised rounds)

Формат: триггер → причина → как избегать.

## 2026-09-19

### 1. Тренер s7 молча падал ~1×/ран, чекпойнт 5500 обрезан (55 МБ вместо 615 778 297)
- Триггер: тихий exit 1 без traceback в PowerShell-логе; запись model_*.pt обрывается.
- Причина: **диск G: был заполнен до 0 байт**. Полный traceback найден в train_log3.txt:
  `torch.serialization RuntimeError: [enforce fail at inline_container.cc:659] unexpected pos ...` —
  запись zip-контейнера .pt падает, когда нет места. 1.45 ТБ диска, _models 1145 ГБ + LMStudio 130 ГБ и пр.
- Как избегать: перед каждым запуском тренера — проверка `Get-PSDrive G` (порог ≥ 20 ГБ).
  Чистка 19.09: удалены superseded merged (auk_ru_18000/22000, s2_A/B_250/500, s5_6000, s6_5000/6000)
  → 51.4 ГБ свободно. Целостность ckpt проверяется размером: 615 778 297 байт.

### 2. dtype RuntimeError "mat1 and mat2 must have the same dtype, but got Float and BFloat16" (train_log5/6)
- Триггер: ручной relaunch s7 без флага `--mixed_precision bf16`.
- Причина: я запускал accelerate с нуля по памяти, а не по эталонному скрипту. bf16_transformer=True
  кастует transformer в BF16, но без accelerate mixed_precision autocast не согласует входные тензоры
  (time_embed) — падение на первом же шаге после resume.
- Как избегать: **запуск тренера только копированием команды из существующего run_s*.ps1** (все содержат
  `--mixed_precision bf16`), не по памяти. Ошибка повторена дважды (log5, log6) до сверки — нарушение
  правила «одна ошибка дважды».

### 3. Aniemore RESD-контроль: 65/150 клипов n/a
- Триггер: emotion_gate.py на RESD-манифесте — почти половина строк без оценки.
- Причина: LABEL_MAP не содержал identity-классы Aniemore (happiness/sadness/enthusiasm):
  RESD-лейблы уже в схеме Aniemore, маппинг был только для наших label'ов (happy/sad/...).
- Как избегать: при использовании маппинга проверять полный словарь лейблов датасета против ключей
  LABEL_MAP до запуска (инвариант: n/n_rows == 0). Исправлено: identity-записи добавлены,
  перезапуск дал 150/150, acc 0.9933.

### 4. TypeError: Object of type float32 is not JSON serializable (повторено дважды)
- Триггер: json.dump метрик, где np.float32 (numpy/soundfile/onnx) не приведены к float().
  Случай 1: rerank_composite.py (исправлен); случай 2: s10_candidate_mine.py — ПОСЛЕ уже
  внесённой записи, т.е. нарушение правила «та же ошибка дважды». Спас rescore из wav
  (s10_rescore.py) без повторной генерации (99 wav переиспользованы).
- Как избегать: в любом скрипте, пишущем метрики в JSON, оборачивать ВСЕ числовые значения
  во float() в момент сборки строки; перед тяжёлым прогоном делать dry-run json.dumps на
  одной строке результата.
