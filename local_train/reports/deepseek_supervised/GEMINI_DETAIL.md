# Слепой набор: детально по вариантам и операциям (EXPLORATORY, авто-сигнал)

Источник: v2 (все группы) + v3 (clone, исправленная семантика утечки). Варианты вскрыты
через SECRET_map ПОСЛЕ слепой оценки. Это не решение: слепое прослушивание опционально.
Кросс-задачных средних для ВЫБОРА не используется; ниже — по-задачные/по-операционным и парные разницы.

## Группа: tts

| вариант | valid/total | overall | text_fidelity | naturalness | voice_similarity_to_ref | accent | stress |
|---|---:|---:|---:|---:|---:|---:|---:|
| u10000 | 48/48 | 3.88 | 4.96 | 5.85 | 8.12 | 9.56 | 8.83 |
| A@250 | 48/48 | 6.23 | 7.5 | 6.48 | 8.17 | 9.65 | 9.42 |
| A@500 | 48/48 | 5.81 | 7.04 | 6.38 | 7.83 | 9.4 | 9.19 |
| B@250 | 48/48 | 5.4 | 6.54 | 6.06 | 8.0 | 9.48 | 8.73 |
| B@500 | 48/48 | 6.35 | 7.71 | 6.52 | 7.9 | 9.44 | 9.4 |

## Группа: phonetics

| вариант | valid/total | overall | text_fidelity | naturalness | voice_similarity_to_ref | accent | stress |
|---|---:|---:|---:|---:|---:|---:|---:|
| u10000 | 16/16 | 6.31 | 7.69 | 6.44 | 8.06 | 9.5 | 9.12 |
| A@250 | 16/16 | 7.38 | 9.31 | 6.94 | 8.19 | 9.81 | 9.5 |
| A@500 | 16/16 | 8.06 | 9.25 | 7.81 | 8.56 | 9.81 | 9.69 |
| B@250 | 16/16 | 8.0 | 9.25 | 7.62 | 8.38 | 9.81 | 9.62 |
| B@500 | 16/16 | 7.88 | 9.31 | 7.5 | 8.38 | 9.56 | 9.5 |

## Группа: phrase_training

| вариант | valid/total | overall | text_fidelity | naturalness |
|---|---:|---:|---:|---:|
| u10000 | 1/1 | 4.0 | 6.0 | 4.0 |
| A@250 | 1/1 | 9.0 | 10.0 | 8.0 |
| A@500 | 1/1 | 8.0 | 10.0 | 7.0 |
| B@250 | 1/1 | 7.0 | 10.0 | 6.0 |
| B@500 | 1/1 | 6.0 | 8.0 | 5.0 |

## Группа: clone

| вариант | valid/total | overall | text_fidelity | voice_similarity_to_ref | naturalness |
|---|---:|---:|---:|---:|---:|
| u10000 | 25/25 | 4.72 | 5.8 | 8.08 | 6.36 |
| A@250 | 25/25 | 7.04 | 8.52 | 8.16 | 6.8 |
| A@500 | 25/25 | 7.96 | 9.44 | 8.48 | 7.44 |
| B@250 | 25/25 | 6.24 | 7.4 | 8.0 | 6.24 |
| B@500 | 25/25 | 7.4 | 8.68 | 8.36 | 7.04 |

## Группа: tool

| вариант | valid/total | overall | operation_performed | content_preserved | quality |
|---|---:|---:|---:|---:|---:|
| u10000 | 35/35 | 6.97 | 7.03 | 8.77 | 8.2 |
| A@250 | 35/35 | 6.51 | 6.46 | 8.17 | 7.94 |
| A@500 | 35/35 | 6.4 | 6.46 | 8.49 | 7.6 |
| B@250 | 35/35 | 6.54 | 6.91 | 8.4 | 8.0 |
| B@500 | 35/35 | 6.14 | 5.86 | 9.57 | 8.97 |

## Группа: capability

| вариант | valid/total | overall | operation_performed | content_preserved | quality |
|---|---:|---:|---:|---:|---:|
| u10000 | 12/12 | 2.33 | 1.92 | 5.92 | 5.42 |
| A@250 | 12/12 | 2.33 | 2.08 | 5.33 | 5.0 |
| A@500 | 12/12 | 3.0 | 2.42 | 6.17 | 5.92 |
| B@250 | 12/12 | 2.42 | 2.42 | 5.5 | 4.83 |
| B@500 | 12/12 | 2.83 | 2.58 | 5.08 | 5.5 |

## По-операционные таблицы: tool
### tool_noise_add

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 0.8 | 0.0 | 0.0 | 8.0 | 0.4 |
| A@250 | 5/5 | 0.6 | 0.0 | 0.0 | 6.4 | 0.8 |
| A@500 | 5/5 | 0.2 | 0.2 | 0.0 | 8.0 | 0.8 |
| B@250 | 5/5 | 0.4 | 2.0 | 0.0 | 6.0 | 0.8 |
| B@500 | 5/5 | 0.6 | 0.0 | 0.0 | 8.0 | 0.8 |

### tool_pitch_down

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| A@250 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| A@500 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| B@250 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| B@500 | 5/5 | 6.6 | 6.0 | 0.6 | 10.0 | 0.0 |

### tool_pitch_up

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 8.2 | 8.0 | 0.8 | 10.0 | 0.0 |
| A@250 | 5/5 | 8.2 | 8.0 | 0.8 | 10.0 | 0.0 |
| A@500 | 5/5 | 8.4 | 8.0 | 0.8 | 10.0 | 0.0 |
| B@250 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| B@500 | 5/5 | 8.2 | 8.0 | 0.8 | 10.0 | 0.0 |

### tool_speed_down

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 5.8 | 7.0 | 0.4 | 7.2 | 0.6 |
| A@250 | 5/5 | 6.8 | 7.2 | 0.6 | 7.2 | 0.4 |
| A@500 | 5/5 | 5.6 | 6.6 | 0.4 | 6.4 | 0.6 |
| B@250 | 5/5 | 5.8 | 6.8 | 0.4 | 6.4 | 0.6 |
| B@500 | 5/5 | 4.4 | 4.0 | 0.4 | 10.0 | 0.0 |

### tool_speed_up

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 4.0 | 4.2 | 0.2 | 6.2 | 0.8 |
| A@250 | 5/5 | 3.6 | 4.0 | 0.2 | 5.6 | 0.8 |
| A@500 | 5/5 | 2.2 | 2.4 | 0.0 | 5.0 | 1.0 |
| B@250 | 5/5 | 3.4 | 3.6 | 0.2 | 6.4 | 0.6 |
| B@500 | 5/5 | 6.8 | 7.0 | 0.6 | 9.0 | 0.2 |

### tool_volume_down

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| A@250 | 5/5 | 8.0 | 8.0 | 0.8 | 8.0 | 0.0 |
| A@500 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| B@250 | 5/5 | 6.2 | 6.0 | 0.6 | 10.0 | 0.0 |
| B@500 | 5/5 | 8.2 | 8.0 | 0.8 | 10.0 | 0.0 |

### tool_volume_up

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| A@250 | 5/5 | 8.4 | 8.0 | 0.8 | 10.0 | 0.0 |
| A@500 | 5/5 | 8.4 | 8.0 | 0.8 | 10.0 | 0.0 |
| B@250 | 5/5 | 10.0 | 10.0 | 1.0 | 10.0 | 0.0 |
| B@500 | 5/5 | 8.2 | 8.0 | 0.8 | 10.0 | 0.0 |

## По-операционные таблицы: capability
### cap_deaccent

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 1.0 | 0.0 | 0.0 | 3.0 | 1.0 |
| A@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |

### cap_emotion_angry

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 1.0 | 0.0 | 9.0 | 0.0 |
| A@250 | 1/1 | 3.0 | 2.0 | 0.0 | 9.0 | 0.0 |
| A@500 | 1/1 | 9.0 | 9.0 | 1.0 | 10.0 | 0.0 |
| B@250 | 1/1 | 8.0 | 8.0 | 1.0 | 9.0 | 0.0 |
| B@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |

### cap_emotion_happy

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 8.0 | 8.0 | 1.0 | 9.0 | 0.0 |
| A@250 | 1/1 | 2.0 | 1.0 | 0.0 | 9.0 | 0.0 |
| A@500 | 1/1 | 4.0 | 3.0 | 0.0 | 9.0 | 0.0 |
| B@250 | 1/1 | 2.0 | 1.0 | 0.0 | 9.0 | 0.0 |
| B@500 | 1/1 | 5.0 | 5.0 | 0.0 | 9.0 | 0.0 |

### cap_emotion_sad

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 1.0 | 0.0 | 10.0 | 0.0 |
| A@250 | 1/1 | 6.0 | 7.0 | 1.0 | 6.0 | 1.0 |
| A@500 | 1/1 | 2.0 | 0.0 | 0.0 | 9.0 | 0.0 |
| B@250 | 1/1 | 4.0 | 4.0 | 0.0 | 10.0 | 1.0 |
| B@500 | 1/1 | 7.0 | 9.0 | 1.0 | 7.0 | 0.0 |

### cap_enhancement

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 1.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 2.0 | 3.0 | 0.0 | 4.0 | 1.0 |
| A@500 | 1/1 | 2.0 | 3.0 | 0.0 | 4.0 | 1.0 |
| B@250 | 1/1 | 2.0 | 3.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 1.0 | 1.0 | 0.0 | 4.0 | 1.0 |

### cap_nonverbal_breath

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| A@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 2.0 | 2.0 | 0.0 | 4.0 | 1.0 |

### cap_pitch_up

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 4.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| A@500 | 1/1 | 1.0 | 0.0 | 0.0 | 10.0 | 0.0 |
| B@250 | 1/1 | 2.0 | 2.0 | 0.0 | 3.0 | 1.0 |
| B@500 | 1/1 | 2.0 | 1.0 | 0.0 | 4.0 | 1.0 |

### cap_quality

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 1.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 3.0 | 2.0 | 0.0 | 5.0 | 1.0 |
| A@500 | 1/1 | 2.0 | 1.0 | 0.0 | 3.0 | 1.0 |
| B@250 | 1/1 | 1.0 | 2.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 2.0 | 2.0 | 0.0 | 4.0 | 1.0 |

### cap_speech_edit

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 2.0 | 0.0 | 3.0 | 1.0 |
| A@250 | 1/1 | 2.0 | 3.0 | 0.0 | 3.0 | 1.0 |
| A@500 | 1/1 | 2.0 | 3.0 | 0.0 | 3.0 | 1.0 |
| B@250 | 1/1 | 2.0 | 3.0 | 0.0 | 3.0 | 1.0 |
| B@500 | 1/1 | 1.0 | 0.0 | 0.0 | 3.0 | 1.0 |

### cap_speed_12

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 3.0 | 3.0 | 0.0 | 6.0 | 1.0 |
| A@250 | 1/1 | 4.0 | 5.0 | 0.0 | 7.0 | 1.0 |
| A@500 | 1/1 | 9.0 | 9.0 | 1.0 | 10.0 | 0.0 |
| B@250 | 1/1 | 4.0 | 6.0 | 0.0 | 8.0 | 1.0 |
| B@500 | 1/1 | 9.0 | 9.0 | 1.0 | 10.0 | 0.0 |

### cap_vocal_extraction

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 2.0 | 2.0 | 0.0 | 4.0 | 1.0 |
| A@250 | 1/1 | 2.0 | 2.0 | 0.0 | 5.0 | 1.0 |
| A@500 | 1/1 | 2.0 | 1.0 | 0.0 | 4.0 | 1.0 |
| B@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 2.0 | 2.0 | 0.0 | 4.0 | 1.0 |

### cap_whisper

| вариант | valid/total | overall | operation_performed | expected_effect_matches | content_preserved | introduced_artifacts |
|---|---:|---:|---:|---:|---:|---:|
| u10000 | 1/1 | 1.0 | 0.0 | 0.0 | 10.0 | 0.0 |
| A@250 | 1/1 | 1.0 | 0.0 | 0.0 | 5.0 | 1.0 |
| A@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@250 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |
| B@500 | 1/1 | 1.0 | 0.0 | 0.0 | 4.0 | 1.0 |

## Фонетические замены (tts+phonetics+phrase)

| вариант | файлов с sub>0 | всего sub | sub/файл |
|---|---:|---:|---:|
| u10000 | 6 | 6 | 0.09 |
| A@250 | 6 | 6 | 0.09 |
| A@500 | 8 | 8 | 0.12 |
| B@250 | 6 | 7 | 0.11 |
| B@500 | 6 | 8 | 0.12 |

## Утечка содержимого референса (clone)

| вариант | leak rate |
|---|---:|
| u10000 | 0.4 |
| A@250 | 0.08 |
| A@500 | 0.0 |
| B@250 | 0.16 |
| B@500 | 0.12 |

## Парные разницы по task_id (значение варианта минус u10000)

| группа | метрика | сравнение | mean delta | лучше/хуже/равно |
|---|---|---|---:|---|
| tts | overall | A@500 - u10000 | 1.94 | 33/5/10 |
| tts | overall | B@500 - u10000 | 2.48 | 37/5/6 |
| phonetics | overall | A@500 - u10000 | 1.75 | 10/2/4 |
| phonetics | overall | B@500 - u10000 | 1.56 | 9/2/5 |
| clone | overall | A@500 - u10000 | 3.24 | 17/2/6 |
| clone | overall | B@500 - u10000 | 2.68 | 17/2/6 |
| tool | overall | A@500 - u10000 | -0.57 | 3/7/25 |
| tool | overall | B@500 - u10000 | -0.83 | 6/10/19 |
| capability | overall | A@500 - u10000 | 0.67 | 2/2/8 |
| capability | overall | B@500 - u10000 | 0.5 | 3/4/5 |

## Худшие файлы (overall, по группам и вариантам; top-5)
- **tts / u10000**: tts34_prosody_alt_male(0151.wav=0), tts02_clusters_alt_male(0285.wav=0), tts38_baseline_alt_male(0303.wav=0), tts42_hard_words_alt_male(0311.wav=0), tts22_numerals_alt_male(0664.wav=0)
- **tts / A@500**: tts10_soft_sign_alt_male(0159.wav=0), tts22_numerals_alt_male(0178.wav=1), tts18_numerals_alt_male(0217.wav=1), tts28_names_user_male(0527.wav=1), tts05_clusters_nat_female(0047.wav=2)
- **tts / B@500**: tts10_soft_sign_alt_male(0447.wav=1), tts15_reduction_alt_female(0098.wav=2), tts18_numerals_alt_male(0454.wav=2), tts22_numerals_alt_male(0427.wav=3), tts27_names_alt_female(0550.wav=3)
- **phonetics / u10000**: ph03_nat_female(0239.wav=2), ph09_nat_female(0091.wav=3), ph15_nat_female(0186.wav=3), ph01_nat_female(0037.wav=5), ph06_user_male(0526.wav=5)
- **phonetics / A@500**: ph12_user_male(0106.wav=5), ph07_nat_female(0096.wav=6), ph03_nat_female(0572.wav=6), ph05_nat_female(0018.wav=7), ph04_user_male(0122.wav=7)
- **phonetics / B@500**: ph05_nat_female(0676.wav=5), ph00_user_male(0089.wav=6), ph10_user_male(0281.wav=6), ph06_user_male(0331.wav=6), ph07_nat_female(0545.wav=6)
- **clone / u10000**: clone02(0260.wav=0), clone16(0328.wav=0), clone01(0586.wav=0), clone17(0131.wav=1), clone13(0256.wav=1)
- **clone / A@500**: clone06(0262.wav=5), clone10(0073.wav=6), clone03(0283.wav=6), clone04(0457.wav=6), clone17(0459.wav=6)
- **clone / B@500**: clone18(0233.wav=3), clone17(0653.wav=4), clone22(0334.wav=5), clone19(0452.wav=5), clone06(0229.wav=6)
- **tool / u10000**: tool_noise_add_0(0343.wav=0), tool_noise_add_4(0375.wav=0), tool_noise_add_2(0043.wav=1), tool_noise_add_3(0492.wav=1), tool_pitch_up_0(0525.wav=1)
- **tool / A@500**: tool_noise_add_0(0400.wav=0), tool_noise_add_4(0507.wav=0), tool_noise_add_3(0536.wav=0), tool_noise_add_1(0644.wav=0), tool_speed_up_2(0102.wav=1)
- **tool / B@500**: tool_noise_add_0(0212.wav=0), tool_noise_add_2(0368.wav=0), tool_speed_down_0(0566.wav=0), tool_noise_add_1(0087.wav=1), tool_pitch_up_0(0180.wav=1)
- **capability / u10000**: cap_whisper(0446.wav=1), cap_deaccent(0485.wav=1), cap_nonverbal_breath(0576.wav=1), cap_vocal_extraction(0001.wav=2), cap_quality(0044.wav=2)
- **capability / A@500**: cap_nonverbal_breath(0181.wav=1), cap_deaccent(0313.wav=1), cap_pitch_up(0381.wav=1), cap_whisper(0393.wav=1), cap_quality(0415.wav=2)
- **capability / B@500**: cap_speech_edit(0202.wav=1), cap_whisper(0265.wav=1), cap_emotion_angry(0300.wav=1), cap_enhancement(0380.wav=1), cap_deaccent(0399.wav=1)

