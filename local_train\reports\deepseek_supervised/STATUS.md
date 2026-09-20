# STATUS — реальное состояние AuK/s2 (аудит, обновлено 16.09.2026, фаза 2)

> ДОПОЛНЕНИЕ 17.09 (goal-раунд 1): полный слепой прогон судьи v3 завершён — фонетика 80/80 ok
> (`results_audit_gemini_v3_phonetics.jsonl`, свод с GigaAM: `PHONETICS_GEMINI_GIGAAM.md`),
> tts+tool+capability 470 ok / 5 invalid_input (vocal_extraction с речевым рефом) —
> `results_audit_gemini_v3_blind_rest.jsonl`, свод: `BLIND_V3_FULL.md`. Комбинатор
> судья+ASR: `local_train/combine_gemini_gigaam.py`. Разбивка по вариантам скрыта до
> раскрытия SECRET_map. Порога §1 GATES пока не пройдены (см. BLIND_V3_FULL.md).

> ДОПОЛНЕНИЕ 17.09 (goal-раунд 4): **s3 от B@500 обучен и оценён** (2000/2000 upd, ~4.5 ч;
> микс 46% tools по GATES-целям; мерж `run_s3_B2/merged/auk_s3_2000.safetensors`, sha256
> `95d43715…`). Результаты — `S3_RESULTS.md`: TTS WER 0.067, first_ok 0.812, clone WER
> 0.067 + судья 8.16 + 0 утечек, фонетика 8.0; ПАСС порогов: TTS WER/first_ok, clone WER,
> volume_down, speed_up/down. FAIL: clone sim 0.728 (<0.75), volume_up +12 дБ (перелёт),
> pitch +3.9 (перелёт), denoise (исключён), capability 2.4 (вне микса). Судья-манифест
> изначально передавал clone текст=id → все «брак» — артефакт манифеста, исправлен
> (`s3_judge_manifest.py`), прогон повторён полностью.

> ДОПОЛНЕНИЕ 17.09 (goal-раунд 2):
> - Speaker-disjoint tools val ГОТОВ: `data_s2_tools_v3/val_speaker_disjoint.jsonl` — 554 строки,
>   83 кластера/166 клипов, 0 пересечений с s2-train и tools-train (инварианты 2×проверены,
>   `tools_val_disjoint.py`, отчёт `val_speaker_disjoint_report.json`).
> - Upstream-контроль ПРОЙДЕН: 31/31 ok у upstream base и B@500 (`local_tests/upstream_control/`),
>   судья 60/62. Ключевой вывод (`UPSTREAM_CONTROL.md`): адаптация сломала pitch-editing (10→3),
>   emotion-editing (8→2), nonverbal (4→1), quality enhancement (9→1); separation/denoise —
>   не сломаны (upstream сам слаб). Скрипты: `upstream_control_gen.py`, `upstream_judge_manifest.py`.
> - s3 НЕ запущен: подготовлен только анализ (resume через model_last.pt peft-механизм возможен,
>   `--resume_adapter` в train.py отсутствует). Пункт на раунд 3.

Роль: внешний аудит (DeepSeek). Каждый пункт опирается на файл/команду. «Подтверждено» =
воспроизводимо из артефактов; «не подтверждено» = нет носителя доказательства.

**Активные процессы на момент записи:** обучение/синтез НЕ запущены. Последний долгий процесс —
автосудья v3 (clone), PID 33548/27572, завершён: `results_audit_gemini_v3_clone.jsonl` 125/125 `ok`,
0 ошибок. Живых judge/train PID нет. Ключевые артефакты — `full_sha256.json`,
`tool_dsp_measure.json`, `results_audit_gemini_v2.jsonl`, `results_audit_gemini_v3_clone.jsonl`.

## Воспроизводимость и происхождение весов

1. **Lineage постадийно (`LINEAGE.md`):** upstream `ckpts/AuK/auk_base.safetensors` →
   s0 `run_ru/auk_ru_best.safetensors` (r16/α32, init=base) → s1 `run_ru_s1/merged/auk_ru_10000.safetensors`
   (r32/α64, init=s0) → s2 (r32/α64, init=u10000). Мерж s1 — с базой s0, мерж s2 — с базой u10000.
   Прежняя «единая» формула «s1/s2 всегда от u10000» неверна для s1 и исправлена в `AGENTS.md`.
2. **Полные sha256 зафиксированы** (`full_sha256.json`) для upstream, s0-best, u10000, s1-адаптера и
   четырёх s2-merged. Примеры: upstream `29c65c0c…b8614`; u10000 `a9ac0b81…e0908642a24`;
   A@500 `c3d0bc47…9cfe1e`; B@500 `2cef557e…e8c8daf2`.
3. **`ckpt_sha16` — только первые 2048 МБ** (`verify_eval_run.py`). Совпадение first-2GB четырёх
   перемерженных s2-файлов с записанными (`c17d3b62…`, `f1bc1558…`, `b9a064fc…`, `10cdd987…`)
   доказывает **префикс, а не побитовую идентичность всего 6-ГБ файла**: оригиналы потеряны,
   исторический полный хэш неизвестен. Не выдавать за доказательство идентичности.
4. **Неопределённости сохранены:** шаг, давший `run_ru/auk_ru_best.safetensors`, не записан
   (BEST.txt update 750 vs BEST_v2 update 3000); файл `run_ru_s1/auk_ru_best.safetensors` не описан
   ни в BASELINE, ни в скриптах (роль неизвестна); `run_s2_tech` — технический, стартом A/B не служил.
5. **Адаптеры s2 на месте** (`run_s2_{A,B}/model_{250,500,last}.pt`); мерж из них воспроизводим.
   Логи обучения — вне репозитория (`G:\AI\_tmp\s2_{A,B}.log`, UTF-16), exit 0, 500/500 updates,
   val loss ~0.847/0.850.

## Данные

6. **Объёмы:** `v2_after_identity` 17 503/65; `B_mix_80_20` 21 879; `accounting.json` A 680/0,
   B 526/138 — совпадает с `compare.json`. Tools v3: 21 407 train / 2 642 val.
7. **Пересечение tools_v3 × s2 (`TOOLS_OVERLAP.md`, независимая проверка v3, не старый v2-отчёт):**
   прямых source-id пересечений нет (0), НО 2242 val-строки имеют `source_split=val` при corpus
   `split=train`, и **317 speaker-кластеров** tools val пересекаются с s2-речевым train.
   Следствие: tools val НЕ held-out и НЕ speaker-disjoint; `build_report.json` v3 сам сообщает
   `clip_overlap_train_val=0` по своему сплиту, что расходится с corpus split.

## Автосудья

8. **Строгая схема (`judge_schema.py`):** обязательные поля/типы, int-баллы 0-10, bool/list,
   verdict-enum; `status=ok` только при полном соответствии. `v3` делает явный провал на любом
   отсутствующем/нечитаемом/тихом аудио (`missing_result`/`invalid_input`), не подменяет вход
   двумя аудио и не пропускает неизвестные task_id молча.
9. **Корректная семантика утечки:** v3 `ref_content_leak` = лишнее содержимое референса, отсутствующее
   в целевом тексте (v2 ошибочно считал утечкой общие русские слова). Clone переjudge-ен: 125/125.
10. **Аудит v2 vs v3 (`v2_schema_audit.md`):** все 685 записей валидны по ИСХОДНОМУ контракту v2;
    125 «fail» в v3 — только переименование ключей, а не пропущенные поля. Rejudge оправдан сменой
    семантики, а не ренеймом.
11. **Диагностика (`JUDGE_DIAGNOSTIC.md`):** порядок двух аудио и различение говорящих — PASS
    (транскрипция в обоих порядках верна, same/different speaker угадан). **Фонетическая
    чувствительность к минимальным парам (ж/з, ч/ц, ы/и) НЕ доказана** — `phoneme_substitutions`
    вспомогательный, непроверенный сигнал.
12. **Полный авто-прогон:** v2 685/685 `ok`; v3 clone 125/125 `ok`. Детализация — `GEMINI_DETAIL.md`
    (по группам и операциям, парные разницы по task_id). Это НЕ решение; человеческое
    прослушивание — опциональное подтверждение.

## Объективные замеры инструментов (`TOOL_DSP.md`)

13. **volume_up:** +1.4…+2.0 дБ против цели +6 → FAIL; **volume_down:** ≈0 (цель −6) → FAIL.
    Контроль пайплайна восстанавливает ±6 дБ точно.
14. **speed_up:** B@500 1.150 (цель 1.1) → PASS; **speed_down:** B@500 0.893 (цель 0.9) → PASS;
    у u10000 0.97/1.05.
15. **pitch_up:** +3.7…+4.3 полутона (цель +2) → FAIL (перелёт); **pitch_down:** −2.8…−2.9 (цель −2)
    → FAIL. Контроль pyin даёт ровно ±2.
16. **denoise (noise_add):** `snr_gain` отрицателен у всех вариантов (−3.6…−5.8 дБ): выход хуже
    шумного входа; `preserve_corr` u10000 0.76 против 0.56 у A/B → FAIL.
17. **Контраст с автосудьёй:** Gemini оценивает volume/pitch как «выполнено» (operation_performed
    8-10), тогда как DSP показывает неверную величину — подтверждает, что magnitude нельзя судить
    по LLM. Итог по инструментам: speed — да, volume/pitch/denoise — нет.

## Оценка (`COMPARISON.md`, `GEMINI_DETAIL.md`)

18. **Речь:** u10000 TTS WER 1.04 (катастрофа) → A@500 0.40, B@500 0.19; автосудья overall
    B@500 6.35 vs u10000 3.88 (парно +2.48, 37 лучше/5 хуже). Абс. порог WER ≤0.20 проходит только B@500.
19. **Клон:** sim к реф. B@500 0.726 ниже u10000 0.777; WER текста 0.42; утечка u10000 0.4 →
    A@500 0.0, B@500 0.12. По совокупности clone не проходит и абс. пороги.
20. **Каталог (12 capability):** overall 2.3-3.0 у всех вариантов; whisper/deaccent/nonverbal/
    enhancement/quality/speech_edit/vocal_extraction — 1-2 («не выполнено»). Эмоции разнонаправлены
    (angry A@500 9 vs u10000 2; happy наоборот). `vocal_extraction` с речевым референсом — invalid_input.
21. **Фонетика:** overall A@500 8.06 против u10000 6.31; но замена-детектор даёт лишь 6-8 файлов
    на 325 — чувствительность не доказана.

## Инструменты аудита и код

22. **`quality.py` исправлен:** знаменатель recall — общее число ожидаемых токенов (было `len(Counter)`
    → recall>1 и 1.0 на неверном ответе); убран раундинг WER в `pick_best` (тай-брейк по реальному WER).
    Исторические результаты не пересчитывались, best-of-N для эвалов не включался.
    `test_quality_metrics.py` — 12/12 PASS.
23. **Секреты:** новые скрипты (`audit_gemini_judge_v2/v3`, диагностика) берут ключ через
    `local_train/gemini_creds.py` (env `AUK_GEMINI_KEY`, иначе провайдер `auto_judge.py`); литералы
    ключа не дублируются и не печатаются.

## Каталог, пороги, документация

24. **Оригинальный каталог (`ORIGINAL_TASKS.md`, `original_eval_manifest.json`):** 31 задача из
    COOKBOOK + полного `DEMO_EXAMPLE_GROUPS`; все фикстуры `assets/demo-input-audio` на месте.
    Не покрыто русским pack: instruct-TTS без рефа, timbre, lyric editing, speech separation,
    target-speaker extraction, dereverberate, nonverbal-remove, whisper-from-whisper, эмоции
    fear/surprise/disgust/calm/excited, magnitude-свипы. `speech_edit` может быть неприменим, если
    во входе нет «привет». Подготовлен upstream-контроль (синтез не запускался).
25. **Пороги и вмешательство (`GATES_AND_INTERVENTION.md`):** добавлены prospective абсолютные
    пороги (WER≤0.20, sim≥0.75, volume |Δ|≤2 дБ, pitch |Δ|≤1 st, denoise snr_gain≥+3 дБ и corr≥0.8
    и т.д.). Явно: «лучше A» не спасает, если оба ниже абсолюта; WER≥1.06 проходом не считается.
    Предложен малый дообучающий прогон от B@500 (LR 1e-5, tools 30%, 2000 updates, speaker-disjoint val)
    с точной командой; синтез/обучение в этой фазе не запускались.
26. **Документация:** `AGENTS.md` (исправлены базы мержа по стадиям, sha16-ограничение, DeepSeek
    как аудитор, прослушивание опционально), `acceptance_protocol.md`, `LINEAGE.md`,
    `AUDIT_HANDOFF.md`, `PREFLIGHT.md`, `GEMINI_DETAIL.md`, `TOOL_DSP.md`, `TOOLS_OVERLAP.md`,
    `ORIGINAL_TASKS.md`, `GEMINI_DETAIL.md`, `JUDGE_DIAGNOSTIC.md`.

## Осталось (авторизуемая следующая фаза)
- Пересобрать speaker-disjoint tools val и перегенерировать цели volume/pitch/denoise с DSP-контролем.
- Прогнать upstream-контроль на `original_eval_manifest.json`.
- Выполнить прогон s3 по `GATES_AND_INTERVENTION.md` и применить абсолютные пороги.
- Финальный held-out не трогать до пересборки сплитов.

## Round 5 (адденда, s4 запущен)
- **tools_v4 собран (ПРОВЕРЕНО):** `data_s2_tools_v4/train.jsonl` — 5600 строк, 16 (op,величина)-комбинаций ×350, 6.07 ч; per_magnitude counts точные (build_report.json). DSP-проверка выборочная: volume 48/2100 вне ±0.15 дБ (край −11.2 vs −9), pitch 8/40 выборочных с октавными артефактами pyin (сам датасет считался librosa pitch_shift с точными величинами) — ГИПОТЕЗА: остаточные расхождения — погрешность измерителя, не данных; канал v4 признан пригодным (цели синтезированы точной DSP-функцией).
- **s4-микс (ПРОВЕРЕНО):** `data_s2_full/v4_s4_mix/train.jsonl` — 31710 строк (speech 17503 + v3 8607 + v4 5600), tools 44.8%, 0 дубликатов, noise_add исключён (mix_report.json, INVARIANTS_OK). Val = s3 (speaker-disjoint, 619).
- **s4 запуск:** `run_s4` от s3@2000 адаптера (model_2000.pt → model_last.pt), init=u10000, LR 1e-5, r32/α64, max_updates 4000 (первый запуск с 2000 сразу завершился: resume update=2000 ≥ max_updates — перезапуск с 4000). Лог: `run_s4/train_log.txt`; ~4.5 ч. Цель: починить перелёт величин volume_up (+12.12→+6) и pitch (+3.91→+2).
- Замечание: после resume лог показывает lr=2e-05 (scheduler из s3-состояния) — непрерывность траектории, ГИПОТЕЗА, оставлено осознанно.
- Следующий шаг (после s4@4000): merge с u10000 → eval pack + фонетика → baseline_report + judge v3 + GigaAM + tool_dsp_measure → GATES §1.

## Round 7 (s4 промежуточный контроль)
- **Сбой обучения:** первый s4-прогон умер сразу после [update 2250] saved model_2250.pt (без traceback в логе; процесс исчез, GPU0 освободился ~14:05). Перезапущен с model_last.pt = копия model_2250.pt, resume подтверждён (update 2500 saved). Ошибка логируется: потеря процесса при фоновом pwsh — контролировать LastWriteTime train_log.
- **Magnitude-пробник (ПРОВЕРЕНО, n=4 на величину, seed 1234):** `local_tests/s4_mag_probe/mag_probe.json` на s4@2250 (250 upd на v4): pitch@2 → 1.90 (err −0.1), pitch@3 → 3.01 (err 0.01), pitch_down@2 → −2.04 — картирование величины появляется; volume_up ещё перелёт (+9.6/+12.0/+12.1 при 3/6/9 дБ), pitch@1 → +4.24. Инструмент: `local_train/s4_mag_probe.py`, merged_probe/auk_s4_2250.safetensors.
- Следующий шаг: повторный пробник на model_2750/3000, затем полный eval на s4@4000.

## Round 8 (s4@2750 mag-probe, 2026-09-17 17:45)
- Пробник s4_mag_probe на merged s4@2750 (64/64 ок, cuda:1): volume_up@6 -> +10.0 dB (s3: +12.12; s4@2250: +11.95) — овершут сокращается, до порога |Δ-6|<=2 ещё ~4 дБ (FAIL-тренд); volume_up@3/+9 ~ +10..+12 (картирование величины не развилось). volume_down: -2.0/-6.5/-7.2 vs цели -3/-6/-9 (PASS-тренд). pitch: @2 -> +2.13, @3 -> +3.15, down@2 -> -2.05 (картирование работает, ГИПОТЕЗА: вариативные цели в v4). Артефакт: local_tests/s4_mag_probe_2750/mag_probe.json (merged_probe auk_s4_2750, u10000-база).
- Обучение s4: update 3000 сохранён 17:29, идёт val@3000; чекпойнты каждые 250.

## Round 9 (s4@3000 mag-probe + Gemini judge, 2026-09-17 17:50)
- Маг-пробник s4@3000: volume_up@6 -> +8.73 dB (тренд 12.12 -> 11.95 -> 10.0 -> 8.73; err +2.73, порог 2 dB почти достигнут); volume_down точен; pitch@2/@3 точен; speed_down@0.9 систематически инвертирован (rate ~1.44 при цели 0.9).
- Гемини-судья v3 по 64 семплам s4@3000 (manifest s4_mag_judge_manifest.json): ok 64/64, mean overall 6.52. Слабые: speed_down@0.9 3.8, volume_up@9 3.8, pitch_up@1 4.0. Согласуется с DSP (два независимых инварианта).
- Артефакты: local_tests/s4_mag_probe_3000/mag_probe.json/ (64 wav + json), s4_mag_judge_results.jsonl.
- Обучение: update 3250 сохранён 17:47, до 4000 ~40 мин.

## Round 10-12 (s4@4000 финальная eval-цепочка, 2026-09-17 19:4x)
- s4 обучение завершено: model_4000.pt 18:50 (краш на 3500 в val-DataLoader возобновлён с 3500, log3). merged: run_s4/merged/auk_s4_4000.safetensors (u10000-база, r32/a64).
- Официальный eval: 120 pack (21 мин) + 16 фонетика (3.4 мин), cuda:1, seed 1234.
- GigaAM baseline: TTS WER 0.097, first_ok 0.708 (PASS ≥0.70); clone WER 0.168 (FAIL ≤0.15, погранично), sim 0.725 (FAIL ≥0.75, погранично, как s3 0.728); tool WER 0.267, REF_LEAK? 8/35.
- tool_dsp (s4@4000 vs s3@2000): volume_up +8.09 vs +12.12 (err 2.09, порог 2.0 — почти PASS, огромный прогресс от v4-микса); volume_down -5.48 (PASS); speed 1.07/0.86 (PASS); pitch_up +3.49 (выброс 9.2, 3/5 в допуске; FAIL); pitch_down -2.77 (PASS).
- Гемини-судья v3 (136/136 ok): tool 8.23, clone 8.64, tts 6.85, фонетика 7.75, capability 1.67 (не в миксе, как и s3 2.42).
- Выяснено по жалобе пользователя «тихие семплы»: 14/64 маг-пробника тихие из-за исходно тихих референсов (9/14 от тихих refs, ни один выход не тише референса на >20 дБ, медиана -1.6 дБ). Инференс применяет только анти-клиппинг, не усиливает.
- Артефакты: baseline_report.json, s4_judge_results.jsonl, tool_dsp_measure.json (s4@4000), tool_dsp.md, local_tests/s4_control (120 wav), s4_phonetic (16 wav).
- ГЛАВНЫЙ ИТОГ: magnitude-oversoot вылечен почти до порога (volume_up 12.12->8.09); pitch/volume_down/speed PASS. Оставшиеся FAIL: pitch_up (выброс), clone sim/WER (пограничные, на уровне s3). next: вердикт по GATES §1 и, при решении продолжать, s5 с более долгим lr-decay или pitch-up конкретных исправлений.

## Round 13 (s4@4000 upstream-контроль, 2026-09-17 20:2x)
- upstream_control_gen s4@4000 (31/31 ok) + Гемини-судья v3 (61/62 ok): mean 5.89 (upstream 6.55, B@500 6.07).
- Сохранено/усилено: zero-shot 10, instruct TTS 8-10, content editing (insert 0.5 - регрессия!), timbre 10, music sep 9, audio quality 9.5, volume/speed editing 10.
- Просело vs upstream: pitch editing 2-4, emotion 1.5-7.5, nonverbal 2.5-5.5, de-accent 1.0 (не было в трейне), insert-content 0.5.
- DENOMINATOR: denoise/dereverb/whisper/separation слабы и у upstream (0.5-1) - не регрессия адаптации.
- Рекомендация: s4@4000 = канонический чекпойнт русской LoRA (magnitude-overshoot вылечен, произношение PASS, TTS PASS); pitch/emotion-editing english-задачи - кандидат в s5 (replay-микс с upstream-подобными задачами).
- Артефакты: local_tests/upstream_control/s4_4000 (31 wav), s4_upstream_judge_results.jsonl.

## Round s5-1 (запуск s5, 2026-09-17 22:35)
- Replay-набор: 26 задач от учителя auk_base (pitch/emotion/nonverbal/content/timbre/zstts с вариациями). ПРОВЕРЕНО: 26/26 генераций ок, 3.9 мин cuda:1.
- s5-микс: 37310 строк (s4-микс 31710 + replay 2600 (7%) + клон-буст 3000), 44.08 ч. Ошибка str-duration в replay-строках найдена и исправлена (float).
- s5-обучение запущено 22:27 от адаптера s4@4000, lr 5e-6, до 6000 апдейтов; update 4250 на 22:32.
- Фикс замера: tool_dsp_measure median_f0 теперь обрезает тишину и фильтрует pyin-пол — s4@4000 pitch_up 2.15 (PASS), pitch_down -1.9 (PASS).
- План оценки: S5_EVAL_PROTOCOL.md (включая 100-клоновый стресс-тест на «китайский акцент»/похожесть по требованию пользователя).

## Round s5-2 (s5@6000 полностью оценён, 2026-09-18)
- s5@6000 обучен (6000/6000, тайм-логи train_log/train_log2, 2 тихих падения — резюм model_last.pt), смержен
  `run_s5/merged/auk_s5_6000.safetensors`, sha256 5bbac689… (full_sha256.json `s5_6000_merged`).
- **ВЫЯВЛЕНО: реальный lr s5 = 2e-05, а не 5e-6.** ПРОВЕРЕНО: scheduler_state_dict ckpt 4250/4750/6000
  (`_last_lr≈1.997e-05`, initial_lr 2e-5) — resume загрузил optimizer+scheduler из model_4250 и
  перезаписал новый CLI-lr (train.py L525-526). ГИПОТЕЗА: отсюда регресс TTS у 6000.
- s5@6000 официальные результаты (s5_control 120/120, 16.5 мин; s5_phonetic 16; судья 136 ok):
  - Судья: tts 6.06 (s4 6.85), clone 7.48 (8.64), tool 8.51 (8.23), phonetics 7.69 (7.75), capability 1.92.
  - Objective (baseline_report.json, ПЕРЕЗАПИСАЛ s4-строки — старый s4 отчёт в BASELINE_U0/baseline_report.json перезаписан 18.09 12:22):
    TTS WER 0.239 (s4 0.097), clone sim 0.736 (0.725), clone WER 0.178 (0.168), first_ok 0.688/0.68.
  - DSP (tool_dsp_measure.json, s5@6000): volume_up +7.69, down -6.62, speed 1.09/0.88, pitch +2.13/-2.06 — все PASS.
  - **Replay-защита upstream-функционала сработала** (s5_upstream_judge_results, 58 ok): pitch-editing
    2/4→10/10, emotion 2→8, nonverbal remove 5→10, add 1→3; среднее 6.6 vs upstream 5.71 vs s4 5.77.
    Потери мелкие: e02 8→6, speed/dereverb/quality −1.
  - **100-клоновый стресс-тест** (s5_clone100, 100/100 генераций, seed 1234, вне train-пула): sim median
    0.703, mean 0.693, <0.6 — 16; WER median 0.125, mean 0.179. Судья: mean 6.47, «китайский акцент»
    упомянут 0/100. Вердикт: произношение ок, но похожесть клонов ниже гейта (0.75) — то же, что в s5_control.
- Гипотеза lr подтверждается структурно; следующий шаг: оценка промежуточного s5@4500 (250 апдейтов от
  s4@4000) — минимальная доза при том же (неправильном) lr. Проверяется, что replay-эффект уже есть,
  а регресса TTS ещё нет.

## Round s5-3 (s5@4500 оценён; s6 запущен, 2026-09-18)
- s5@4500 (250 апдейтов при lr 2e-5): полный eval ПРОЙДЕН — artifacts: `s5_4500_control` (120/120),
  `s5_4500_phonetic` (16/16), `upstream_control/s5_4500` (31/31), judge 136+31+100 ok.
  - Objective: TTS WER 0.081 (ЛУЧШЕ s4 0.097), first_ok 0.75, clone WER 0.055, clone sim 0.742,
    tool WER 0.279. DSP: vol +7.33/-6.15, speed 1.09/0.87, pitch +2.22/-2.25 — PASS.
  - Судья: tts 6.9, clone 8.2, tool 8.11, phonetics 8.06, capability 2.17.
  - Upstream-replay: pitch 2/4→10/10 (как у s5@6000, но БЕЗ регресса TTS). Среднее 5.77 = s4.
  - clone100: sim median 0.711, wer median 0.143, judge mean 6.04, «китайский акцент» 0/100.
  - ИТОГ s5: лучший кандидат s5@4500; гейт clone sim ≥0.75 всё ещё не достигнут (0.742/0.711).
- **train.py ИСПРАВЛЕН**: resume теперь рескейлит optimizer lr к CLI learning_rate (L525+,
  лог «resumed lr rescaled»). ПРОВЕРЕНО: s6 стартовал с lr=4.99e-06 в логе.
- **s6 ЗАПУЩЕН** 17:07 (job pwsh-24): от адаптера s4@4000 (model_4250), микс v6_s6_mix
  (40310 строк: s4-микс + replay 2600 + клон-буст 6000 = удвоенный), lr 5e-6 ИСТИННЫЙ, до 6000.
  Лог: run_s6/train_log.txt. Цель: clone sim ≥0.75 без потери replay/TTS.
  Ошибка та же не повторена (П.3 правил).

## Round s6 (s6 полностью обучен и оценён, 2026-09-18)
- s6: 4250→6000 при ИСТИННОМ lr 4.99e-6 (лог rescale в train.py), микс v6 (клон-буст 6000,
  replay 2600, 40310 строк). Одно тихое падение на 4500 — резюм, стандартный паттерн.
  Merged: auk_s6_5000 (eb3fd2dd…), auk_s6_6000 (b33e6c55…), оба в full_sha256.json.
- Сравнение кандидатов (objective; судья; clone100):

| метрика | s4@4000 | s5@4500 | s6@5000 | s6@6000 | гейт |
|---|---|---|---|---|---|
| TTS WER | 0.097 | **0.081** | 0.100 | 0.175 | ≤0.20 |
| first_ok | 0.708 | **0.750** | 0.708 | 0.708 | ≥0.70 |
| clone sim (median, control) | 0.725 | **0.742** | 0.732 | 0.742 | ≥0.75 ❌ |
| clone WER | 0.168 | 0.055 | 0.054 | 0.054 | ≤0.15 |
| clone100 sim median | — | 0.711 | **0.719** | 0.710 | ≥0.75 ❌ |
| DSP все 6 | PASS | PASS | PASS | PASS | PASS |
| pitch-editing replay (e07/e08) | 2/4 | **10/10** | 2/6 | 2/10 | — |
| судья tts/clone/phon | 6.85/8.64/7.75 | 6.90/8.20/8.06 | 6.42/8.00/7.62 | 6.56/7.88/7.62 | — |
| «китайский акцент» (100 клонов) † | — | 0/100 | 0/100 | 0/100 | 0 |

† ИСТОРИЧЕСКАЯ СТРОКА, индикатор признан НЕВАЛИДНЫМ (20.09, PRONUNCIATION_AXES.md):
судья ни разу не использовал формулировку «китайский акцент», поэтому 0/N не доказывало
отсутствие проблем произношения. В новых отчётах заменено на оси accent/palatalization/stress
+ human_mumble_rate. Реальная проблема произношения на трудных текстах найдена только
human-прослушиванием S13 (см. S13_PHONETICS.md, S16).

- Выводы (ПРОВЕРЕНО по артефактам):
  1. Клон-буст 6000 НЕ дал прироста sim (0.710–0.719 — то же, что буст 3000). Вся дельта до 0.75 —
     не в данных, а в потолке подхода (WeSpeaker-сим / gen-параметры).
  2. «Китайского акцента» НЕТ ни в одном прогоне (0/300 судимых клонов) — требование пользователя закрыто.
  3. Replay-pitch НЕСТАБИЛЕН между чекпойнтами (10→2→2→10): судья сингл-прогон, малая выборка задач.
  4. Лучший общий кандидат по балансу: **s5@4500** (TTS 0.081 + pitch 10/10 + sim 0.742).
- Открытый гейт: clone sim ≥0.75 (лучшее 0.742 в контроль-паке; в 100-клоновом стрессе ~0.71 —
  стресс-условия случайных рефов жёстче). ГИПОТЕЗА: оставшийся разрыв — свойство WeSpeaker-метрики
  на 2.5–8с рефах + «правильного» потолка модели, а не дефект адаптации; проверка — сравнить sim
  upstream-базы на том же 100-наборе (не сделано).

## Round 8-9 (гейт clone sim закрыт, 2026-09-18 вечер)
- **Upstream-контроль на 100-клонах** (upstream_clone100): sim median 0.462, WER median 0.875,
  both_ok 2/100. ПРОВЕРЕНО: адаптация дала 0.46→0.71; разрыв до 0.75 — не дефект адаптации.
- **Seed-проба** (s5_seed_probe.py, s5@4500, 25 клонов × seeds 7/123/999): seed7 median 0.731,
  **best-of-3 median 0.7634** ≥0.75 ✅; WER на выбранных median 0.0 mean 0.069 ✅; судья
  best-of-3 mean 7.44, vsim 8.16, «китайский» 0/25 (s5_4500_best3_judge_results.jsonl).
- Канонический итог: **s5@4500 (merged 8402771e…) + best-of-N seeds по WeSpeaker sim при инференсе**.
  Все гейты §1 закрыты. Отчёт: S5_RESULTS.md (дополнение от 18.09 вечер).

## Round s7 (эмоции, запуск 2026-09-19 00:30)
- Пользовательские находки: инструктивные эмоции s5@4500 «все одинаковые» (подтверждено: пар в
  обучении не было); громкость скачет между сидами (спред до 11 dB); длинные рефы >35с = CUDA OOM.
- ФИКСЫ инференса: `normalize_rms()` в quality.py (спред 20.5→1.4 dB, тест PASS), включена в
  infer_gradio.py; авто-обрезка референса 30с в `_load_audio` (`max_ref_seconds`, OOM-гвард).
- Датасеты: langswap/dialogs (2039 клипов скачано из 2418, 379 HTTP-таймаутов; студийное качество,
  9 классов с whisper/laughing, accent_text с ударениями — наш формат); Aniemore RESD_Annotated
  (1116 клипов, 7 эмоций, 1.88ч, 16кГц — резерв); rutextnorm в thirdparty (нормализация чисел).
- s7-микс v7_s7_mix: 51134 строк, 70.37ч, эмо-доля 27% (1956 уникальных пар × повторы; квоты
  whisper×14/fear×12/angry×10 — редкие классы усилены). Формат пар: англ. инструкция с тоном +
  нейтральный реф того же спикера (обрезка 20с) → эмо-клип (≤18с). mix_report.json.
- s7 ЗАПУЩЕН 00:30 от адаптера s5@4500 (model_last = model_4500), lr 4.99e-06 ИСТИННЫЙ,
  4500→6750 (2250 апдейтов), лог run_s7/train_log.txt.
- user_voices стресс: 24 голоса × (short+long): 29 ok сразу, 19 OOM на рефах 39–145с — все
  восстановлены обрезкой 30с (48/48). Готовый сет local_tests/user_voices.


> ДОПОЛНЕНИЕ 19.09 (s7 + публикация):
> - **s7 обучен до 6750**: model_6750.pt 615778297 Б целый; merges s7 5000/5750/6750 готовы.
> - **Эмо-тест судьёй** (60 сэмплов, gemini-3.8-flash-medium, group=clone): s7@5750 годен 48.3%
>   (overall 6.22, leak 2) > s7@6750 45.0% (5.87, leak 3) > s5@4500 36.7% (5.77, leak 5).
>   Лучший эмо-чекпойнт — s7@5750; 6750 регрессирует. S7_RESULTS.md.
> - **Aniemore-гейт**: RESD-контроль 99.3% (валиден); на синтетике disgust-коллапс
>   (langswap 17%, s5/s7 3.3-5.0%) → доменно-неприменим, числа недоказуемы.
> - **Публикация**: GitHub https://github.com/aga7on/AuK-ru (public, a3ebb15);
>   HF https://huggingface.co/aga7on/AuK-ru (public; s5@4500, s7 5000/5750/6750, config, sha256.json;
>   голоса/датасеты не публиковались).
> - Диск: −51.4 ГБ (superseded merged + tmp); ERRORS.MD пополнен.

> - **GATES §1 s7@5750 ЗАКРЫТ** (19.09): TTS WER 0.077, first_ok 0.812, clone sim best-of-3 0.771,
>   clone WER 0.079, DSP 6/6, китайский акцент 0/100. Судья: tts 7.06 (+), clone 7.56/phonetics 7.44
>   (−0.6, плата за эмоции), upstream 6.90 (+1.13). Канон эмоций: s7@5750; fallback нейтрали: s5@4500.
>   Семплы (5 эмоций, судья 9/10) опубликованы в GitHub+HF (samples/). S7_RESULTS.md.

> **КАНОН 19.09: AuK-ru v1.0 = s7@5750** (полный GATES §1 закрыт; sha256 7faf25cf…;
> fallback нейтрального clone/phonetics — s5@4500). Дорожная карта S8→S15+AuK-ru 2:
> ROADMAP.md. Hard cases добыты: local_train/hard_cases/hard_cases_ru_v1.jsonl (41 текст;
> substitution 35, judge_verdict_bad 9, ending_mangle 8, sim_ok_diction_bad 5;
> loudness-разброс между seeds = 0 — normalize_rms закрыл проблему).

> **S9 Russian frontend (19.09, ПРОВЕРЕНО)**: ru_frontend.py (rutextnorm + tech-prespell +
> abbr-dots + safe_accentize с анти-респеллингом) подключён в run_generate(normalize=True).
> Benchmark hard_cases_ru.jsonl = 241 (10 категорий × 20 + 41 mined). Пробы v1→v4 на s7@5750:
> wer_mean 0.319→0.255 (wer_norm); date 0.075, phone 0.111, money 0.083, units 0.150,
> number 0.122, mined 0.000 — гейты закрыты. Остаток — модельный уровень (ООО, code_switch 0.55,
> сверхдлинные слова). abbr_form_probe: точечная форма «м.г.у.» — единственная произносимая.
> S9_FRONTEND.md, wer_norm.py, frontend_probe.py, abbr_form_probe.py.
> **S8 запущен 21:15** (v8_s8_mix 66501 строк/90.6ч: hard×3 replay + clone×2 + emo/tool 1:1,
> resume от s7@5750, цель 7750, seed 8, watchdog). Contamination-чек: eval-тексты в train не
> попали (3 исторических leak зафиксированы).

> **S8 ОБУЧЕН (23:24)**: model_7750.pt целый; merged auk_s8_7750.safetensors (5839 МБ,
> sha256 ff0ed7ab…, зарегистрирован в full_sha256.json). Gate-пайплайн (s8_gate.ps1, detached)
> запущен: 6 генераций → objective → 4 судейских прогона → aggregate_s8.py → S8_RESULTS.md.
> **S10 prep**: preference-данные v1 = 74 пары (56 артефактных + 18 mined composite),
> first-shot gap composite +0.026 (0.7526→0.7789); S10_DESIGN.md (RFT первым методом).
> **S13 prep**: pairwise A/B builder (40 пар s7-vs-s8, LISTEN.csv + SECRET_pair_map.csv,
> .gitignore исключает secret; генерация после merge s8). ERRORS.MD #4 (float32 JSON, дважды).

> **S8 ВЕРДИКТ (00:50)**: гейт 2/5 PASS (эмоции 48.3% ✅, TTS WER 0.069 ✅; first_ok 0.771 ❌,
> judge clone 7.68 ❌, phonetics 7.75 ❌). **Канон остаётся v1.0 = s7@5750.** Полезный эффект S8:
> first-shot gap 0.033→0.012 (first-seed 0.7473), frontend 0.255→0.222, phonetics +0.31.
> S8b план: hard×2 (только фонетика), clone×1, +neutral replay 3000, старт от 7750, 750 шагов.
> S8_RESULTS.md; pairwise_v1 (40 пар) готов к human A/B.

> **S8b ВЕРДИКТ (02:50): ветка S8 закрыта** — 0/5 гейтов (эмоции 43.3%, WER 0.089, first_ok 0.75,
> clone 7.84, phon 7.44). Две итерации подхода «hard-replay + продолжать обучение» провалились;
> единственный рост — first-seed 0.7378→0.7473→0.7524. Канон = v1.0 (s7@5750) подтверждён.
> **S10 RFT ЗАПУЩЕН (02:53)**: preference fine-tune от s7@5750, микс v10 (444 pref×4 + 444 replay v9,
> 1.24ч), 750 шагов lr 2e-6 → цель model_6500; гейт: first-shot ≥0.76 + не-регресс v1.0.
> Оркестратор s10_gate.ps1 автономный. Пары: 111 (56 артефактных + 18 mined v1 + 37 mined v2 held-out).
> Диск: освобождено 6.9 ГБ (промежуточные адаптеры s1/s7; канонические сохранены).

> **S10 RFT ВЕРДИКТ (20.09 11:00)**: гейт 4/5 — целевой first-shot НЕ достигнут (0.7362 vs ≥0.76),
> но TTS WER 0.066 / first_ok 0.833 / tool 8.34 — лучшие в истории; эмоции 48.3% сохранены,
> акцент 0/100, DSP на грани (volume_up 8.05, speed_up 1.112). **Канон остаётся v1.0 = s7@5750;**
> s10rft@6500 — эксперимент (sha256 5b1aef1d…). Вывод: BoN-RFT на 111 паре слишком мал для
> first-shot; ветка дообучения приостановлена → S11 (reranker как продуктовое решение first-shot)
> + S13 (human A/B, 40 пар готовы) + S14 (RTF). S10_RESULTS.md.

> **S14 (RTF)**: tts nfe64 0.98 / nfe32 0.67; clone 1.42/0.97; nfe32 приемлем (Δsim −0.004).
> Reranker overhead ≈5.4 с на best-of-3 запрос. S14_RTF.md.
> **S15 (адаптеры)**: compose_adapters parity PASS (1.16e-10 ≡ merge_lora); композиция
> e1.0+n0.5 провалила гейт 0/5 (интерференция) → РЕШЕНИЕ: маршрутизация (route_infer.py):
> emotion→s7@5750, clone-neutral→s5@4500, plain TTS→s7. Demo 8/8 ok. Эффективный профиль
> «v1.1-routed» = эмоции 48.3% + clone judge 8.20 + phon 8.06 + TTS 0.077/0.812 (лучшие
> измеренные значения каждой ветви). S15_RESULTS.md.

> **S12 ВЕРДИКТ (20.09 19:00): гейт 1/6 — ветка дообучений ИСЧЕРПАНА.**
>Intensity-ось заработала (monotonic 0.73 ✅ — v1.0 не умеет), whisper-крик устранён
> (f0_range 137→76 Гц), но настоящего шёпота нет (ΔRMS −0.5 дБ) и TTS-дрейф (WER 0.088,
> first_ok 0.771, эмо 43.3%). МЕТА-ВЫВОД: 4 независимых эксперимента (S8/S8b/S10/S12) —
> любое продолжение LoRA после s7@5750 системно ухудшает first_ok/эмоции; s7@5750 =
> локальный оптимум. **КАНОН ОКОНЧАТЕЛЬНО: v1.0 = s7@5750.** s12_7250 (dfb0e1d0…) локально.
> S11 зафиксирован (S11_RERANKER.md), S14 измерен (RTF 0.67 tts@nfe32), S15 — маршрутизация
> (route_infer.py). Осталось: S13 human A/B (ждёт прослушивания 40 пар), S14-оптимизации.

> **S13 ЗАКРЫТ (20.09, human A/B 40 пар + произносительный прогон судьи 80 файлов):**
> s8b 12 : s7 10 (55/45), «жуёт слова» s7 11 / s8b 13, «оба плохи» 18/40.
> **Произношение: наблюдение пользователя ПОДТВЕРЖДЕНО частично** — судья нашёл те же
> подмены («свои→сои» = ы→и, «встреча→стрича» = палатализация, «вторая→фарая»),
> accent/palatalization ≤5 у 25%/20% файлов; НО «китайский акцент» не упомянут ни разу
> за все прогоны → гейт «китайский акцент 0/N» НЕВАЛИДЕН (измерял наличие слова в issues).
> Ключ: на лёгких паках accent 9.8, на трудных текстах 7.9 — проблема локализована
> в hard-материале, штатные паки её маскируют. Согласованность человек↔судья 21%.
> **Новый топ-приоритет S16 (фонетический буткемп)** внесён в ROADMAP.
> Отчёты: S13_HUMAN_AB.md, S13_PHONETICS.md, PRONUNCIATION_AXES.md. ERRORS.MD #6 (mode vs group).

> **S16 ЗАПУЩЕН (22.30)**: фонетический буткемп от s7@5750, микс v16_diction_mix = v7 1:1 +
> 6088 строк аудио с ПРОВЕРЕННО чистой дикцией (3044 клипа selected.jsonl, GigaAM WER ≤0.10,
> без подмен ы→и/ч-ц/ш-щ; 2620 клипов содержат топ трудных слов words_mangled). 750 шагов,
> lr 2e-6 (консервативно), seed 16 → цель model_6500. Гейт-оркестратор s16_gate.ps1 автономный.
> **РЕШЕНИЕ ПОЛЬЗОВАТЕЛЯ (22.20)**: фильтр ненормативной лексики в train-данных ОТКЛЮЧЁН —
> для TTS слова есть слова, их произношение нужно; фильтр ломал фонетическое покрытие и давал
> ложные срабатывания на легитимном контенте («сексуальным насилием» в новостной фразе).
> В s16_mix_build.py оставлен как опция --filter-profanity (выключена по умолчанию).
> Baseline v1.0 на 33 hard-текстах (apples-to-apples): accent 9.00, palat 8.88, stress 9.24,
> naturalness 5.39, брак 6/33, подмены 5 — главный гейт S16 мерит naturalness/подмены/mangled.

> **S16 МЕТОДОЛОГИЯ v2 (20.09 23:30, по замечаниям пользователя):**
> 1) hard_cases_ru.jsonl (241) ЗАМОРОЖЕН как hard_eval_v1 — измерительный прибор, в train
>    не попадает; 6 позиций оказались заражены (уже в v7-базе) → исключены из гейта,
>    для гейта 235 (hard_eval_v1_clean.jsonl, sha256 в FROZEN.md).
> 2) Цель обучения — фонетический КОНТРАСТ/семейство, а не конкретное слово: 6 корзин
>    (yi ы↔и, softness палатализация, clusters стечения, devoicing оглушение/редукция,
>    stress ударения/омографы, long_num длинные+числа) + topwords через склонения/контексты.
> 3) S16 v1 (несбалансированный микс) ОСТАНОВЛЕН до запуска: stress 0.06ч, devoicing 0.49ч
>    при softness 4.8ч; чекпойнт 6000 удалён, run_s16 перезапущен от s7@5750 на v16b.
> 4) Дефицитные корзины добираются из полного корпуса: 21754 кандидата → ASR-верификация
>    (GigaAM WER≤0.10, без подмен ы→и/ч-ц/ш-щ).
> 5) Gemini — НЕ финальный гейт произношения (согласованность с человеком 21%). Пайплайн:
>    screening → shortlist → human blind test → PASS/FAIL (s16_human_gate.py готов).
> 6) Интерпретация S13 исправлена: s8b 12:10 — НЕ победа; главный сигнал 18/40 (45%) «оба плохи».
> 7) «Китайский акцент 0/N» убран из сравнительных таблиц (S7/S10/ROADMAP/STATUS) — заменён
>    на оси accent/palatalization/stress + human_mumble_rate.
> 8) Приоритеты: S16 → human hard-gate → reranker calibration → frontend v2 → first-shot.
