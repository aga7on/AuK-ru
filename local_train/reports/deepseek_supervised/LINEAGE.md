# LINEAGE — происхождение весов по стадиям (16.09.2026, из launch-скриптов и артефактов)

Правило: каждый переход подтверждён launch-скриптом/конфигом, а не общей формулой. Хэши:
полный sha256 — `full_sha256.json`; `ckpt_sha16` — только первые 2048 МБ (`verify_eval_run.py`).

```
AuK upstream  ckpts/AuK/auk_base.safetensors
   full sha256 29c65c0c6045e8d8fb454019f99c9680508f553fc98fa143feaca3711d0b8614
   │
   └─ s0  run_train_bg.ps1:  init=ckpts/AuK/auk_base.safetensors, LoRA r16/α32, out=run_ru
        run_ru/auk_ru_best.safetensors
          full bb63701168eda0138e08fbce0e3a57ff6ee5fa48e252b9cfc5773156a9ae2aa6
          first-2GB 0985a2f9ac6aedb5
        │
        └─ s1  run_train_s1(.resilient).ps1:  init=run_ru/auk_ru_best.safetensors, r32/α64, out=run_ru_s1
             merge base (run_params_ab2.ps1 и др.): run_ru/auk_ru_best.safetensors
             run_ru_s1/merged/auk_ru_10000.safetensors
               full a9ac0b81f5159ce707afc31e4897be2236716163144447092f7a0e0908642a24
               first-2GB 4f3eca357e96c273  (совпадает с u0_control run_params.ckpt_sha16)
             adapter run_ru_s1/model_10000.pt  full b4891ab6bc50982242a0e41e465c846c63dec741c4503a292b3bcf2bf30b2e16
             │
             └─ s2  run_s2_A.ps1 / run_s2_B.ps1:  init=run_ru_s1/merged/auk_ru_10000.safetensors, r32/α64
                  merge base (run_s2_chain/finalize): тот же u10000
                  run_s2_A/merged/auk_s2_A_250.safetensors  full 7785d8d6217ac8a0676cec9467a5e677562c35830b362b5544406588b97e1397
                  run_s2_A/merged/auk_s2_A_500.safetensors  full c3d0bc47413f27e12e8d13bf0ed9f81dc4fe43ade67ba446738cb078049cfe1e
                  run_s2_B/merged/auk_s2_B_250.safetensors  full f794d8e445d968ae2db82558c083f9dcb1d991d5e03a4fdcd07f08946e120ee8
                  run_s2_B/merged/auk_s2_B_500.safetensors  full 2cef557ee9be95c5bb5748c51265b7e57ce3f613923bb1c3177e4f9de8c8daf2
```

## Что доказано
- Каждый init берётся из ЯВНОГО `--init_ckpt` в обёртке (см. `rg init_ckpt local_train/*.ps1`).
- Мержи s2, пересозданные 16.09, совпадают с записанными `ckpt_sha16` (первые 2 ГБ)
  для всех четырёх: A@250 `c17d3b62e895feff`, A@500 `f1bc15582c227b1a`, B@250 `b9a064fcb6d0b28e`,
  B@500 `10cdd9872b0235ea`. **Но это доказывает только префикс 2 ГБ, а не побитовую идентичность
  всего 6-ГБ файла**: оригинальные merged-файлы были потеряны, полный исторический хэш неизвестен.
  Полные sha256 восстановленных файлов зафиксированы и уникальны.

## Неопределённости (сохранить, не «упрощать»)
- `run_ru/auk_ru_best.safetensors`: слаг «best» из `BEST.txt` (update 750) и `BEST_v2.txt`
  (update 3000) указывает на разные шаги; к какому именно update относится сохранённый merged —
  не записано (адаптеров s0 в `run_ru` нет). s0-принадлежность достоверна, номер шага — нет.
- `run_ru_s1/auk_ru_best.safetensors` (first-2GB `8555c10d8f7a1956`) существует, но в
  `BASELINE.md`/launch-скриптах не описан. Роль неясна (не путать с `run_ru/auk_ru_best.safetensors`).
- `run_s2_tech` (init u10000) — технический прогон; стартом A/B не служил (AB_PROTOCOL, п.3).
- `ckpt_sha16` — частичный хэш; для любых будущих сверок фиксировать полный sha256.
