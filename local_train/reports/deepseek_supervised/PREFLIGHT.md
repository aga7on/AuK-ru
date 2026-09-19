# PREFLIGHT — проверка автосудьи Gemini v2 (16.09.2026)

Цель: до любого полного прогона убедиться, что судья (`audit_gemini_judge_v2.py`,
`PROMPT_VERSION=audit-2026-09-16-v2`, alias `gemini-3.8-flash-medium`) слышит аудио,
различает испорченные примеры и не выдумывает успех. Набор сознательно маленький и смешанный.

Артефакты: `preflight_manifest.json`, `preflight_results.jsonl` (полные строки с SHA256 аудио,
prompt_sha256, seconds), `preflight_results.raw.jsonl` (полные raw-ответы), `preflight_controls\*.wav`.

## Состав (12 реальных задач + 5 контролей)

| # | файл (вариант скрыт) | задача | тип | control |
|---|---|---|---|---|
| 1 | 0070.wav | tts00_clusters_user_male | tts | real |
| 2 | 0003.wav | tts28_names_user_male | tts | real |
| 3 | 0089.wav | ph00_user_male | phonetics | real |
| 4 | 0006.wav | ph02_user_male | phonetics | real |
| 5 | 0024.wav | phrase_sber | phrase_training | real |
| 6 | 0051.wav | clone00 | clone | real |
| 7 | 0269.wav | tool_volume_up_0 | tool | real |
| 8 | 0212.wav | tool_noise_add_0 | tool | real |
| 9 | 0137.wav | tool_pitch_up_0 | tool | real |
| 10 | 0282.wav | cap_emotion_happy | capability | real |
| 11 | 0148.wav | cap_speech_edit | capability | real |
| 12 | 0001.wav | cap_vocal_extraction | capability | real |
| 13 | pos_target_phrase.wav | preflight_pos | tts | positive_real_target |
| 14 | neg_silence.wav | preflight_neg_silence | tts | negative_silence |
| 15 | neg_trunc.wav | preflight_neg_trunc | tts | negative_truncated |
| 16 | neg_noise.wav | preflight_neg_noise | tts | negative_noise |
| 17 | neg_wrong_text.wav | preflight_neg_wrong | tts | negative_content_mismatch |

## Результаты

| файл | mode | status | overall | ключевые поля | verdict |
|---|---|---|---|---|---|
| 0070.wav | tts | ok | 7 | fid=8 nat=8 sim=8 sub=[] | доработка |
| 0003.wav | tts | ok | 6 | fid=8 nat=6 sim=8 sub=[] | доработка |
| 0089.wav | phonetics | ok | 6 | fid=10 nat=5 sim=7 sub=[] | доработка |
| 0006.wav | phonetics | ok | 8 | fid=10 nat=7 sim=8 sub=[] | годен |
| 0024.wav | tts | ok | 8 | fid=10 nat=7 sim=8 sub=[] | годен |
| 0051.wav | clone | ok | 7 | fid=10 sim=9 leak=False | доработка |
| 0269.wav | tool | ok | 2 | op=0 match=False cont=10 art=False | брак |
| 0212.wav | tool | ok | 0 | op=0 match=False cont=0 art=True | брак |
| 0137.wav | tool | ok | 10 | op=10 match=True cont=10 art=False | годен |
| 0282.wav | capability | ok | 2 | op=1 match=False cont=10 art=False | брак |
| 0148.wav | capability | ok | 2 | op=2 match=False cont=3 art=True | брак |
| 0001.wav | capability | ok | 2 | op=2 match=False cont=4 art=True | брак |
| pos_target_phrase.wav | tts | ok | 7 | fid=10 nat=6 sim=7 sub=[] | годен |
| neg_silence.wav | tts | ok | 0 | fid=0 nat=0 sim=0 sub=[] | брак |
| neg_trunc.wav | tts | ok | 0 | fid=0 nat=0 sim=0 sub=[] | брак |
| neg_noise.wav | tts | ok | 2 | fid=8 nat=2 sim=3 sub=[] | брак |
| neg_wrong_text.wav | tts | ok | 0 | fid=0 nat=5 sim=2 sub=[] | брак |

## Проверка чувствительности (positive / negative manipulation)

| контроль | что сделано | overall | ожидание | итог |
|---|---|---|---|---|
| positive_real_target | pos_target_phrase.wav | 7 | высокая точность | PASS |
| negative_silence | neg_silence.wav | 0 | overall низко, truncated | PASS |
| negative_truncated | neg_trunc.wav | 0 | overall низко, truncated | PASS |
| negative_noise | neg_noise.wav | 2 | artifacts низко, overall низко | PASS |
| negative_content_mismatch | neg_wrong_text.wav | 0 | text_fidelity низко | PASS |

## Raw-ответы контролей (полностью, из `preflight_results.raw.jsonl`)

### positive_real_target — pos_target_phrase.wav
```json
{"audio_access": true, "text_fidelity": 10, "endings": 10, "naturalness": 6, "prosody": 6, "accent": 10, "palatalization": 10, "stress": 10, "artifacts": 9, "voice_similarity_to_ref": 7, "truncated": false, "words_mangled": [], "words_dropped": [], "words_misstressed": [], "phoneme_substitutions": [], "issues": "Фраза звучит несколько монотонно и с искусственными паузами между словами, однако дикция четкая и без речевых ошибок.", "overall": 7, "verdict": "годен"}
```
### negative_silence — neg_silence.wav
```json
{"audio_access": true, "text_fidelity": 0, "endings": 0, "naturalness": 0, "prosody": 0, "accent": 0, "palatalization": 0, "stress": 0, "artifacts": 0, "voice_similarity_to_ref": 0, "truncated": true, "words_mangled": [], "words_dropped": ["Сбер", "включи", "музыку", "для", "пробежек", "через", "десять", "минут"], "words_misstressed": [], "phoneme_substitutions": [], "issues": "В аудиозаписи синтеза полностью отсутствует речь (тишина).", "overall": 0, "verdict": "брак"}
```
### negative_truncated — neg_trunc.wav
```json
{"audio_access": true, "text_fidelity": 0, "endings": 0, "naturalness": 0, "prosody": 0, "accent": 0, "palatalization": 0, "stress": 0, "artifacts": 0, "voice_similarity_to_ref": 0, "truncated": true, "words_mangled": [], "words_dropped": ["Сбер", "включи", "музыку", "для", "пробежек", "через", "десять", "минут"], "words_misstressed": [], "phoneme_substitutions": [], "issues": "АУДИО 2 отсутствует. Предоставлена только одна аудиозапись с посторонней речью референса, целевой текст синтеза не озвучен.", "overall": 0, "verdict": "брак"}
```
### negative_noise — neg_noise.wav
```json
{"audio_access": true, "text_fidelity": 8, "endings": 8, "naturalness": 2, "prosody": 3, "accent": 9, "palatalization": 8, "stress": 9, "artifacts": 1, "voice_similarity_to_ref": 3, "truncated": false, "words_mangled": [], "words_dropped": [], "words_misstressed": [], "phoneme_substitutions": [], "issues": "Поверх всей дорожки наложен громкий непрерывный электрический гул/треск; голос звучит неестественно, роботизированно и торопливо.", "overall": 2, "verdict": "брак"}
```
### negative_content_mismatch — neg_wrong_text.wav
```json
{"audio_access": true, "text_fidelity": 0, "endings": 0, "naturalness": 5, "prosody": 5, "accent": 10, "palatalization": 8, "stress": 8, "artifacts": 8, "voice_similarity_to_ref": 2, "truncated": false, "words_mangled": [], "words_dropped": ["жёлтый", "жук", "жужжит", "над", "розами", "уже", "два", "часа"], "words_misstressed": [], "phoneme_substitutions": [], "issues": "В синтезе воспроизведён совершенно другой текст ('Сбер, включи музыку для пробежек через 10 минут'), целевая фраза полностью отсутствует.", "overall": 0, "verdict": "брак"}
```

## Оценка судьи

- Контроли пройдены: real target fidelity=10, максимум overall по отрицательным = 2 (silence/trunc/wrong=0, noise=2, artifacts=1).
- Судья различает тишину, обрыв, шум и несовпадение текста — значит чувствительность есть.
- Валидный JSON и `audio_access=true` во всех 17 строках; `cap_vocal_extraction` с первого раза
  вернул `audio_access=false` и был пере-запрошен политикой ретраев (112 сек) — ужесточить
  обработку долгих зависаний заметно нужно, поэтому таймаут и повтор оставлены.
- `naturalness`/`prosody` у реального человеческого таргета = 6, а не 10: шкала не завышена,
  «10» достижимо только для идеала. Учитывать при сравнении абсолютных средних.
- `operation_performed`/`expected_effect_matches` дают конкретику по инструментам
  (например, `tool_volume_up_0`: op=0 → громкость не изменилась; `tool_noise_add_0`:
  op=0, cont=0, introduced_artifacts=true → грубый сбой; `tool_pitch_up_0`: op=10).

## Ограничения и вывод

- Это ОДИН слепой сэмпл на задачу: выводы о вариантах (A/B/u10000) делать нельзя, только о судье.
- Для `cap_vocal_extraction` референс — речевой `user_ref.wav`, а не смесь с музыкой; тест
  методически некорректен (см. AUDIT_2026-09-15.md). Результат по нему в отчётность не берём.
- Magnitude (dB/полутона) судья не измеряет; нужен отдельный DSP-замер (см. acceptance_protocol).
- Заключение: автосудья v2 пригодна как ВСПОМОГАТЕЛЬНЫЙ сигнал после правок; решение о вариантах
  принимается по слепому прослушиванию и объективным замерам. Полный прогон — после фиксации
  merge-весов (STATUS.md п.1-2).
