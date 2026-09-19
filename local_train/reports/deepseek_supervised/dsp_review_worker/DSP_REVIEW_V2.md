# DSP review v2 — независимый перезамер инструментов s2

Worker: DeepSeek v4.1 Flash (DSP correction), bounded run. Источник: `dsp_core_v2.py` / `dsp_measure_v2.py`. Файлы-артефакты: 175 WAV (7 операций × 5 задач × 5 вариантов). Сырые записи: `dsp_v2_perfile.json`, агрегат: `dsp_v2_summary.json`, входные пути+sha256: `dsp_v2_inputs.json`.

## Что исправлено относительно `tool_dsp_measure.py`

1. **Активность:** абсолютный энергетический порог (dBFS) + порог относительно p95; all-zero/non-finite/too-short → жёстко invalid (тишина больше не «активна»). Абсолютный уровень активной речи (`active_rms_db_in/out`) пишется в per-file JSON, near-silent выходы (< −55 dBFS) флагуются — это вскрыло `tool_pitch_down_0` (все варианты ≈−66 dBFS), который старый скрипт считал валидным.
2. **Denoise:** сегментный SNR/корреляции только после явного выравнивания (ограниченный лаг ±300 мс) и проверки применимости по корреляции огибающих. При фазовом/временном расхождении — `inconclusive`, а не число. Добавлена фазо-устойчивая вторичная метрика (снижение шумового пола в нерабочих кадрах чистого эталона). Тишина не считается идеальным denoise. SR-мисматч — явный resample с флагом, без молчаливого обрезания.
3. **Громкость:** FLOAT-WAV с запасом, медиана/IQR покадрового гейна, peak и clip ratio; контроли с положительными И отрицательными ожиданиями.
4. **Скорость:** прокси по активному **span** + явный гейт сопоставимости контента (stretch-инвариантный MFCC nearest-neighbour + DTW); если контент несопоставим — invalid. **Питч:** парные voiced-кадры (а не голое отношение медиан), voiced coverage, octave-guard.
5. **Чистый эталон** берётся из происхождения датасета (`noise_add_val_<id>.wav`, проверено равенство kyutai-клипу), пути и sha256 логируются.
6. **Агрегация:** n/valid/invalid по op×variant, распределение эффекта, перспективные пороги; среднее НЕ объявляется вердиктом.

## Контроли пайплайна: 16/16 asserted

| контроль | ожидание | asserted |
|---|---|---|
| pos_volume_up_6db | volume_up on exactly +6 dB -> valid and |delta-(6.0)|<=0.6 | PASS |
| pos_volume_down_6db | volume_down on exactly -6 dB -> valid and |delta-(-6.0)|<=0.6 | PASS |
| pos_speed_1.1 | speed_up at rate 1.1 -> valid, comparable, |rate-1.1|<=0.05 | PASS |
| pos_speed_0.9 | speed_down at rate 0.9 -> valid, comparable, |rate-0.9|<=0.05 | PASS |
| pos_pitch_2.0 | pitch_up by +2 st -> valid, |semitones-(2.0)|<=0.8, no octave flag | PASS |
| pos_pitch_-2.0 | pitch_down by -2 st -> valid, |semitones-(-2.0)|<=0.8, no octave flag | PASS |
| pos_denoise_perfect | clean ground truth as output -> applicable and snr_gain_db>0 | PASS |
| neg_vol_silence | all-zero output -> invalid, no gain reported | PASS |
| neg_vol_nonfinite | NaN output -> invalid nonfinite | PASS |
| neg_vol_unchanged | 0 dB change on volume_up -> valid measurement but outside +6 window, no direction | PASS |
| neg_vol_wrongdirection | wrong direction (-6 dB measured as volume_up) -> direction_ok False | PASS |
| neg_speed_dropped_words | speed-up with a dropped middle chunk -> NOT a clean speed pass (invalid or flagged) | PASS |
| neg_speed_delay | 150 ms delay is not a speed change -> rate~1.0 (|rate-1|<=0.05) | PASS |
| neg_denoise_delay | delayed noisy copy -> alignment recovers ~150 ms and snr_gain stays near 0 (no fabricated denoise) | PASS |
| neg_sr_mismatch | 16 kHz output vs 24 kHz source -> resampled flag, gain ~0 (no silent truncation) | PASS |
| neg_denoise_zero | zero ground truth/output -> invalid, never counted as perfect denoise | PASS |

Полные наблюдения контролей: `dsp_v2_controls.json`.

## Перспективные пороги (объявлены до прогона, не подогнаны)

- volume: up (3.0, 9.0) dB (цель +6), down (-9.0, -3.0) dB (цель −6)
- speed: up (1.0, 1.25) (цель 1.1), down (0.8, 1.0) (цель 0.9)
- pitch: up (0.8, 3.5) st (цель +2), down (-3.5, -0.8) st (цель −2)
- denoise: применимо при corr огибающих ≥ 0.6 и |лаг| ≤ 300.0 мс; вторично снижение шумового пола ≥ 1.0 dB

## Результаты по операциям (знаменатель n=5 на вариант)

### volume_up — метрика `delta_db` (dB)

| вариант | valid/5 | delta_db median [min..max] | dir ok/valid | in window | clip out | invalid | worst |
|---|---:|---|---:|---:|---:|---|---|
| u10000 | 5/5 | 0.907 [0.325..2.116] | 4/5 | 0/5 | 0 | — | tool_volume_up_3@u10000=0.325 |
| A@250 | 5/5 | 2.602 [0.644..3.37] | 5/5 | 1/5 | 0 | — | tool_volume_up_3@A@250=0.644 |
| A@500 | 5/5 | 2.376 [0.352..3.397] | 4/5 | 1/5 | 0 | — | tool_volume_up_3@A@500=0.352 |
| B@250 | 5/5 | 1.819 [0.618..3.146] | 5/5 | 1/5 | 0 | — | tool_volume_up_3@B@250=0.618 |
| B@500 | 5/5 | 2.156 [0.698..4.745] | 5/5 | 2/5 | 0 | — | tool_volume_up_3@B@500=0.698 |

### volume_down — метрика `delta_db` (dB)

| вариант | valid/5 | delta_db median [min..max] | dir ok/valid | in window | clip out | invalid | worst |
|---|---:|---|---:|---:|---:|---|---|
| u10000 | 5/5 | 0.083 [-0.965..1.508] | 1/5 | 0/5 | 0 | — | tool_volume_down_4@u10000=1.508 |
| A@250 | 5/5 | 0.123 [-0.937..1.499] | 2/5 | 0/5 | 0 | — | tool_volume_down_4@A@250=1.499 |
| A@500 | 5/5 | 0.049 [-1.134..1.378] | 2/5 | 0/5 | 0 | — | tool_volume_down_4@A@500=1.378 |
| B@250 | 5/5 | -0.142 [-1.157..0.854] | 2/5 | 0/5 | 0 | — | tool_volume_down_4@B@250=0.854 |
| B@500 | 5/5 | -0.098 [-1.115..0.965] | 2/5 | 0/5 | 0 | — | tool_volume_down_4@B@500=0.965 |

### speed_up — метрика `rate_proxy` (x)

| вариант | valid/5 | rate_proxy median [min..max] | dir ok/valid | in window | invalid | worst |
|---|---:|---|---:|---:|---|---|
| u10000 | 4/5 | 1.083 [0.839..1.103] | 3/4 | 3/4 | content_not_comparable_rate_proxy_unreliable×1 | tool_speed_up_3@u10000=0.839 |
| A@250 | 3/5 | 1.095 [0.843..1.097] | 2/3 | 2/3 | content_not_comparable_rate_proxy_unreliable×2 | tool_speed_up_3@A@250=0.843 |
| A@500 | 3/5 | 1.095 [0.843..1.097] | 2/3 | 2/3 | content_not_comparable_rate_proxy_unreliable×2 | tool_speed_up_3@A@500=0.843 |
| B@250 | 3/5 | 1.098 [1.006..1.103] | 3/3 | 3/3 | content_not_comparable_rate_proxy_unreliable×2 | tool_speed_up_3@B@250=1.006 |
| B@500 | 5/5 | 1.095 [1.081..1.139] | 5/5 | 5/5 | — | tool_speed_up_2@B@500=1.081 |

### speed_down — метрика `rate_proxy` (x)

| вариант | valid/5 | rate_proxy median [min..max] | dir ok/valid | in window | invalid | worst |
|---|---:|---|---:|---:|---|---|
| u10000 | 0/5 | — | 0/0 | 0/0 | content_not_comparable_rate_proxy_unreliable×5 | tool_speed_down_4@u10000=-0.068 |
| A@250 | 0/5 | — | 0/0 | 0/0 | content_not_comparable_rate_proxy_unreliable×5 | tool_speed_down_4@A@250=-0.069 |
| A@500 | 0/5 | — | 0/0 | 0/0 | content_not_comparable_rate_proxy_unreliable×5 | tool_speed_down_4@A@500=-0.07 |
| B@250 | 0/5 | — | 0/0 | 0/0 | content_not_comparable_rate_proxy_unreliable×5 | tool_speed_down_4@B@250=-0.037 |
| B@500 | 5/5 | 0.901 [0.802..1.016] | 4/5 | 4/5 | — | tool_speed_down_2@B@500=1.016 |

### pitch_up — метрика `semitones` (st)

| вариант | valid/5 | semitones median [min..max] | dir ok/valid | in window | invalid | worst |
|---|---:|---|---:|---:|---|---|
| u10000 | 5/5 | 2.0 [2.0..2.0] | 5/5 | 5/5 | — | tool_pitch_up_0@u10000=2.0 |
| A@250 | 5/5 | 2.0 [1.9..2.0] | 5/5 | 5/5 | — | tool_pitch_up_3@A@250=1.9 |
| A@500 | 5/5 | 2.0 [1.9..2.0] | 5/5 | 5/5 | — | tool_pitch_up_3@A@500=1.9 |
| B@250 | 5/5 | 2.0 [1.9..2.0] | 5/5 | 5/5 | — | tool_pitch_up_3@B@250=1.9 |
| B@500 | 5/5 | 2.0 [1.9..2.0] | 5/5 | 5/5 | — | tool_pitch_up_3@B@500=1.9 |

### pitch_down — метрика `semitones` (st)

| вариант | valid/5 | semitones median [min..max] | dir ok/valid | in window | invalid | worst |
|---|---:|---|---:|---:|---|---|
| u10000 | 3/5 | -1.9 [-2.0..-1.9] | 3/3 | 3/3 | output_has_no_active_speech×1, voiced_content_not_comparable×1 | tool_pitch_down_2@u10000=-1.9 |
| A@250 | 3/5 | -1.9 [-2.0..-1.9] | 3/3 | 3/3 | output_has_no_active_speech×1, voiced_content_not_comparable×1 | tool_pitch_down_2@A@250=-1.9 |
| A@500 | 3/5 | -1.9 [-2.0..-1.9] | 3/3 | 3/3 | output_has_no_active_speech×1, voiced_content_not_comparable×1 | tool_pitch_down_2@A@500=-1.9 |
| B@250 | 3/5 | -1.9 [-2.0..-1.9] | 3/3 | 3/3 | output_has_no_active_speech×1, voiced_content_not_comparable×1 | tool_pitch_down_2@B@250=-1.9 |
| B@500 | 3/5 | -1.9 [-2.0..-1.9] | 3/3 | 3/3 | output_has_no_active_speech×1, voiced_content_not_comparable×1 | tool_pitch_down_2@B@500=-1.9 |

### noise_add — метрика `snr_gain_db` (dB)

| вариант | valid/5 | applicable | snr_gain_db median [min..max] | noise_floor_red dB | invalid | worst |
|---|---:|---:|---|---:|---|---|
| u10000 | 4/5 | 4 | -6.83 [-9.73..-5.37] | 13.28 | denoise_evidence_inconclusive_alignment_or_groundtruth×1 | tool_noise_add_2@u10000=-9.73 |
| A@250 | 3/5 | 3 | -8.4 [-10.37..-5.35] | 10.39 | denoise_evidence_inconclusive_alignment_or_groundtruth×2 | tool_noise_add_2@A@250=-10.37 |
| A@500 | 3/5 | 3 | -8.51 [-10.48..-5.3] | 10.33 | denoise_evidence_inconclusive_alignment_or_groundtruth×2 | tool_noise_add_2@A@500=-10.48 |
| B@250 | 3/5 | 3 | -8.35 [-10.65..-5.26] | 10.32 | denoise_evidence_inconclusive_alignment_or_groundtruth×2 | tool_noise_add_2@B@250=-10.65 |
| B@500 | 3/5 | 3 | -8.18 [-11.24..-5.59] | 8.65 | denoise_evidence_inconclusive_alignment_or_groundtruth×1, output_has_no_active_speech×1 | tool_noise_add_2@B@500=-11.24 |

## Наблюдения (факты, без вердикта по вариантам)

- **volume_up:** запрошено +6 dB, но медиана по вариантам [0.907, 2.602, 2.376, 1.819, 2.156] dB (в окно (3.0, 9.0) попало [0, 1, 1, 1, 2] из 5). Направление верное, величина сильно недобрана.
- **volume_down:** запрошено −6 dB, медиана [0.083, 0.123, 0.049, -0.142, -0.098] dB, верное направление лишь [1, 2, 2, 2, 2]/5. Худший (B@500): tool_volume_down_4=0.965 dB (рост, а не снижение).
- **speed_down:** сопоставимый контент есть только у B@500 ([0, 0, 0, 0, 5]/5 valid); у остальных прокси ненадёжен → invalid, а не ложный «rate».
- **speed_up:** B@500 — единственный с 5/5 valid; прочие имеют 0-2 несопоставимых.
- **pitch_up:** стабильно ~+2 st во всех вариантах (парные voiced-кадры).
- **pitch_down:** 3/5 valid; `tool_pitch_down_0` у всех вариантов near-silent (≈−66 dBFS), `tool_pitch_down_3` — источник почти без voiced-кадров (coverage 0.14) → invalid.
- **noise_add:** применимость только [4, 3, 3, 3, 3]/5; где применимо — сегментный SNR-gain отрицательный (метрика штрафует ресинтез), но фазо-устойчивое снижение шумового пола (медиана, dB) [13.28, 10.39, 10.33, 10.32, 8.65] > 0. Вывод о denoise не делается (inconclusive/weak); худший применимый — tool_noise_add_2=-11.24 dB.

## Denoise — применимость и ограничения

Сегментный SNR считается против чистого клипа. Для регенерированного нейро-аудио это слабая метрика: любое расхождение фазы/просодии/синтеза штрафуется как «шум». Поэтому замер помечается применимым только при хорошем выравнивании, иначе — `inconclusive`. Ни один случай не объявляется доказанным denoise.

- u10000: applicable 4/5, evidence=['inconclusive', 'segmental_snr_applicable']
- A@250: applicable 3/5, evidence=['inconclusive', 'segmental_snr_applicable']
- A@500: applicable 3/5, evidence=['inconclusive', 'segmental_snr_applicable']
- B@250: applicable 3/5, evidence=['inconclusive', 'segmental_snr_applicable']
- B@500: applicable 3/5, evidence=['inconclusive', 'segmental_snr_applicable']

## Контент-доказательства (ASR/Gemini)

Проверенного per-file mapping варианта на строки автосудьи в репозитории нет без открытия `local_tests/blind_s2/SECRET_map.csv`, что запрещено до завершения слепого прослушивания. Поэтому content-evidence не подставлялся. Доступен только агрегат `exploratory/blind_s2_gemini_v2_by_variant.md` (вспомогательный, не решение). Скрипт поддерживает `--content-evidence PATH` для привязки после появления верифицированного mapping.

## Оговорки

- Никакой агрегатный «pass» не объявляется: решение по вариантам требует слепого прослушивания; этот отчёт — объективный magnitude-замер.
- speed — прокси активного span, условный на сопоставимость контента; pitch — парные voiced-кадры; оба помечаются invalid при несопоставимости.
- Один фиксированный seed librosa/pyin; артефакты не перегенерировались.

Всего записей per-file: 175; контролей: 16 (16 asserted).
