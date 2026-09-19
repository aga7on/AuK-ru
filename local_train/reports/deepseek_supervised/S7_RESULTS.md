# S7_RESULTS — эмоциональный этап (судья + гейт) — ПРОВЕРЕНО 19.09

Этап: s7 (микс v7_s7_mix, emo-доля 0.27, teacher-инструкция «Reproduce the reference voice and
say in Russian with a {EMO} tone»), обучение до 6750 (lr 5e-6, r32/α64, resume-цепочка).

## Протокол эмо-теста (одинаков для s5 и s7)
- 5 эмоций (happy, sad, angry, fearful, excited) × 3 фразы × 2 сида × 2 голоса (clone08 male, clone01 female) = 60.
- Генерация: `emotion_ru_test_s7.py` (nfe=64, cfg 2.0, normalize_rms), merged ckpt + config из run_s2_B.
- Судья: Gemini `gemini-3.8-flash-medium` (локальный прокси 8045), v3, group=clone (валидная схема).
- Сэмплы: `local_tests\emotion_ru_s7_5750\` (s7), `local_tests\emotion_ru\` (s5).

## Результаты судьи (60/60 ok в обоих)

| метрика | s5@4500 | s7@5750 | Δ | s7@6750 |
|---|---|---|---|---|
| годен | 22/60 (36.7%) | **29/60 (48.3%)** | +11.6 п.п. | 27/60 (45.0%) |
| доработка | 27 | 20 | −7 | 19 |
| брак | 11 | 11 | =0 | 14 |
| overall | 5.77 | **6.22** | +0.45 | 5.87 |
| text_fidelity | 8.15 | **8.33** | +0.18 | 8.18 |
| voice_similarity_to_ref | 7.18 | **7.22** | +0.04 | 7.00 |
| naturalness | 5.63 | **5.73** | +0.10 | 5.50 |
| artifacts | 6.13 | **6.35** | +0.22 | 6.20 |
| ref_content_leak | 5 | **2** | −3 | 3 |

По эмоциям (годен): s7@5750: happy 7/12, fearful 7/12, excited 6/12, angry 5/12, sad 4/12;
s7@6750: 6/6/6/6/4. Слабое звено у всех — naturalness (5.5–5.7): «запинки, неестественная интонация
на стыке фраз».

**Вывод (ПРОВЕРЕНО судьёй 19.09): s7@5750 — лучший эмо-чекпойнт** (48.3% годен, overall 6.22, leak 2);
s7@6750 регрессирует к концу обучения (брак 14, overall 5.87) — переобучение эмоций/флэттен.
Для канона-MVP эмоций брать s7@5750; полный GATES §1 (клон/TTS-нерегресс) на s7@5750 — следующий этап.

## Aniemore-гейт (проверка эмоций классификатором)

| сет | acc | коллапс |
|---|---|---|
| RESD (контроль, домен обучения гейта) | **99.3%** (149/150) | нет |
| langswap-учитель | 17% (17/100) | disgust 60/100 |
| s5@4500 (наш синтез) | 3.3% (2/60) | disgust ~53/60 |
| s7@5750 (наш синтез) | 5.0% (3/60) | disgust 53/60 |

ГИПОТЕЗА снята / ПРОВЕРЕНО: коллапс в «disgust» у гейта на любом НЕ-RESD-подобном аудио (учитель,
наши генерации) — доменный мисматч WavLM-Aniemore. На синтетическом домене гейт НЕ различает эмоции,
поэтому 3.3%/5.0% не являются доказательством отсутствия эмоций; сравнение корректно только судьёй
и человеческим прослушиванием. Гейт применим к RESD-подобным записям (контроль 99.3%).

## Обучение s7 (факты)
- Дошло до 6750 (model_6750.pt 615 778 297 Б — целый), чекпойнты 4750…6750.
- 5 тихих падений процесса; причина 1–3 — заполненный диск (torch.serialization enforce fail),
  причина 5 — пропущенный `--mixed_precision bf16` при ручном relaunch (моя ошибка, log5/log6).
  С 4-го relaunch — watchdog `watch_s7.ps1` (авторестарт с последнего целого ckpt).
  См. ERRORS.MD.

## Файлы
- `local_tests\emotion_ru_s7_5750\` — 60 wav + results.json + aniemore_results.json
- `local_tests\emotion_ru\` — s5 заново прогнанные (results.json + aniemore_results.json)
- `local_train\reports\deepseek_supervised\emotion_s7_5750_judge_results.jsonl` (60 ok)
- `local_train\reports\deepseek_supervised\emotion_s5_judge_results.jsonl` (60 ok)
- `local_train\run_s7\merged\auk_s7_{5000,5750,6750}.safetensors` (по 6.12 ГБ)
- sha256: full_sha256.json (s7_*_merged)

## Следующий шаг
- Судья + гейт на финальном s7@6750 (тот же протокол) → выбор канона s5@4500 vs s7@6750.
- Полный eval-цикл s7 (eval pack 120, phonetics 16, upstream 31, clone100, DSP) — GATES §1 не-регресс.