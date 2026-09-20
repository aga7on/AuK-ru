# S11 — Quality-aware Inference (фиксация этапа, 20.09.2026)

Статус: **реализован v1, опубликован; калибровка весов ждёт S13 (human A/B).**

## Что сделано
- `local_train/rerank_composite.py` — composite reranker best-of-N:
  score = 0.35·sim + 0.25·asr_fidelity + 0.20·dnsmos + 0.10·loudness + 0.10·pause
  − repetition_penalty − artifact_penalty.
  Все компоненты локальные (WeSpeaker ONNX, GigaAM ASR, speechmos DNSMOS, DSP),
  naturalness = реальный DNSMOS (заглушка 0.5 заменена 19.09).
- `local_train/generate_v1.py` — единая точка входа v1.0-рецепта: frontend → best-of-3
  (seeds 7/123/999) → composite rerank → normalize_rms/limit_peak; TTS-режим без рефа
  скоринг renormalized (asr 0.40/nat 0.30/loud 0.15/pause 0.15). Smoke: TTS/clone/emo ok.
- Smoke-демонстрация целевого поведения (clone01, 3 seeds): выбран seed7 (sim 0.652, ASR 1.00,
  score 0.738) вместо seed123 (sim 0.765, ASR 0.75, score 0.722) — «голосом похож, но
  заговаривает слово» больше не проходит (кейс из мотивации ROADMAP S11).
- RTF-стоимость (S14_RTF.md): rerank ≈1.8 c/кандидат (ASR 1.41 — основной), best-of-3 ≈5.4 c.

## Что осталось (зависит от S13)
- Калибровка весов на human-labelled парах (текущие веса — стартовые из ROADMAP);
- naturalness-прокси: DNSMOS измеряет чистоту, не естественность просодии — после S13
  возможна замена на обученный скорер по парам.

## Артефакты
`rerank_composite.py`, `generate_v1.py` (оба в GitHub), smoke-логи в отчётах S9/S10.