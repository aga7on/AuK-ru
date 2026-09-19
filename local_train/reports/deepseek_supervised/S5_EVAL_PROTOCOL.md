# S5 EVAL PROTOCOL (дообучение от s4@4000, запуск 22:27)

## Цели s5
1. Clone sim ≥ 0.75 (было 0.725), clone WER ≤ 0.15 (было 0.168).
2. Сохранить все s4 PASS: TTS WER ≤0.20, first_ok ≥0.70, volume_down/speed/pitch DSP.
3. Replay-функционал: upstream pitch/emotion/nonverbal/content/timbre не хуже s4 (mean 5.89),
   цель — приблизить к upstream 6.55.
4. Большие генерации: 100 случайных клонов — Гемини-судья на «акцент/похожесть/качество»
   + WeSpeaker sim по каждому; выбор любых голосов из корпуса (вне тренировочного пула).

## Доказательная цепочка (после model_6000)
1. merge s5@6000 → run_s5/merged/auk_s5_6000.safetensors (base u10000, r32/α64) + sha256.
2. eval_pack 120 + phonetic 16 → cuda:1.
3. baseline_report (GigaAM WER + WeSpeaker sim).
4. tool_dsp_measure (VARIANTS + "s5@6000": "s5_control").
5. Гемини-судья v3 манифест 136.
6. upstream_control_gen s5_6000 (31) + судья → сравнение с upstream 6.55 / s4 5.89.
7. NEW: 100-клоновый стресc-тест: 100 случайных рефов ru_wav_mfa (не из train пула
   v2_after_identity — сверка по спикер-хешам/файл-именам), генерация «Reproduce the
   reference voice and say in Russian: '<текст>'», замеры: WeSpeaker sim, GigaAM WER,
   Гемини-судья (акцент: русский нейтральный? голос похож? артефакты?).
   Пороги: sim ≥0.75 медиана, WER ≤0.20, «китайский акцент» = 0 в вердиктах судьи.
8. GATES §1 вердикт + S5_RESULTS.md.

## Стоп-условия
- Если s5 ломает s4-DSP (volume/pitch/speed) → откат на более ранний чекпойнт s5.
- Если клон не улучшается к 6000 → инспекция семплов, вердикт о следующем шаге.
