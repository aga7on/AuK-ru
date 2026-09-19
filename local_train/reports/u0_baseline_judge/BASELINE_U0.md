# Базовое качество u10000 (u0-контроль, первые попытки)

- Дата: 2026-09-16T04:49:22; каталог: `G:\AI\AuK\local_tests\u0_control`
- Протокол: без best-of-N, без трима; NFE=64, seed=1234. Судейский слой — доп. сигнал, не ground truth.
- Критерий удачной первой попытки: WER ≤ 0.15, нет ASR_ERROR/пустого (для capability — вручную).

## Итоги по категориям

| Категория | n | удачных первых (текст) | + без клиппинга | WER mean | WER median | sim median | judge mean |
|---|---|---|---|---|---|---|---|
| tts | 48 | 0.292 | 0.271 | 1.04 | 0.837 | 0.793 | 7.92 |
| clone | 25 | 0.24 | 0.24 | 0.958 | 0.5 | 0.777 | 6.4 |
| tool | 35 | 0.371 | 0.371 | 0.432 | 0.333 | 0.625 | 5.8 |
| capability | 12 | None | None | None | None | 0.873 | 4.82 |

## TTS по голосам (sim vs ref)

| Голос | n | sim median | WER mean |
|---|---|---|---|
| alt_female | 12 | 0.821 | 1.167 |
| alt_male | 12 | 0.834 | 2.324 |
| nat_female | 12 | 0.782 | 0.612 |
| user_male | 12 | 0.762 | 0.058 |

## Худшие 12 по WER

- **clone15** WER=9.00 sim=0.8458 flags=['REF_LEAK?']
  - expected: д+евять, в+осемь с+емь три с+емь, ноль в+осемь в+осемь п+ять, дв+а.
  - heard:    включи телеканал айва аш ди на тв серия два
- **tts10_soft_sign_alt_male** WER=4.00 sim=0.7895 flags=['REF_LEAK?']
  - expected: Сельский пейзаж радовал глаз.
  - heard:    ну потому что для них это прям классическая история это вот к наше я не знаю
- **tts18_numerals_alt_male** WER=3.25 sim=0.8348 flags=['REF_LEAK?']
  - expected: Сорок семь участников зарегистрировались заранее.
  - heard:    ну сорок семь мужей это прям классическая история это вот как наша я не знаю
- **tts02_clusters_alt_male** WER=2.75 sim=0.7874 flags=['REF_LEAK?']
  - expected: Объёмный звук наполнил комнату.
  - heard:    ну потому что для них это прям классическая нату три ботыря
- **tts46_hard_words_alt_male** WER=2.60 sim=0.8324 flags=['REF_LEAK?']
  - expected: Здравствуйте, рады видеть вас снова.
  - heard:    ну потому что для них это прям классическай это вотка на шея незнова
- **tts42_hard_words_alt_male** WER=2.40 sim=0.8135 flags=['REF_LEAK?']
  - expected: Любопытство взяло верх над осторожностью.
  - heard:    ну потому что для них настороженности это вот как наша не знаю
- **tts38_baseline_alt_male** WER=2.29 sim=0.8818 flags=['REF_LEAK?']
  - expected: Завтра утром я отправлю вам подробный ответ.
  - heard:    нупотому что для них это прям классическая отве это вот как как наша я не зря
- **tts05_clusters_nat_female** WER=2.25 sim=0.8109 flags=['REF_LEAK?']
  - expected: Впрыск топлива происходит автоматически.
  - heard:    в прысс топлива порой сходят атомачты они вообще не спят
- **tts14_reduction_alt_male** WER=2.20 sim=0.8523 flags=['REF_LEAK?']
  - expected: Кажется, что дождь скоро закончится.
  - heard:    ну потому что то тоже скоро закончится это вот как наша я не знаю
- **tts22_numerals_alt_male** WER=2.00 sim=0.8096 flags=['REF_LEAK?']
  - expected: Три четверти пути уже позади.
  - heard:    нупотому что для них это прям классическая идея три ботыря
- **tts26_names_alt_male** WER=2.00 sim=0.8811 flags=['REF_LEAK?']
  - expected: Наталья Семёновна ведёт литературный кружок.
  - heard:    наталья семьяних это прям классическая история это вотка наши ри богатыря
- **clone22** WER=2.00 sim=0.5988 flags=['REF_LEAK?']
  - expected: арсен+ал.
  - heard:    арсин ал

## Capability (прослушивание вручную)

- **cap_emotion_happy** sim=0.7334 dur=6.0 flags=[] judge={"text_fidelity": 5, "endings": 10, "naturalness": 10, "prosody": 10, "accent": 10, "palatalization"
  - heard: надо записать наверное голос будет а нет нет все нормально я хочу просто записать голос
- **cap_emotion_sad** sim=0.8158 dur=6.0 flags=[] judge={"text_fidelity": 10, "endings": 10, "naturalness": 10, "prosody": 10, "accent": 10, "palatalization
  - heard: надо записать наверное голос будет нет нет все нормально я хочу просто записать голос
- **cap_emotion_angry** sim=0.7064 dur=6.0 flags=[] judge={"text_fidelity": 10, "endings": 10, "naturalness": 10, "prosody": 10, "accent": 10, "palatalization
  - heard: надо записать наверное голос будет нет нет все нормально я хочу просто записать голос
- **cap_whisper** sim=0.8803 dur=6.0 flags=[] judge={"text_fidelity": 5, "endings": 10, "naturalness": 10, "prosody": 10, "accent": 10, "palatalization"
  - heard: анадо записать наверное голос будет нет нет все нормально я хочу просто записать голос
- **cap_speech_edit** sim=0.8932 dur=6.0 flags=[] judge={"text_fidelity": 0, "endings": 7, "naturalness": 10, "prosody": 9, "accent": 10, "palatalization": 
  - heard: надо записать наверное голос будет нет не я просто зудовости
- **cap_deaccent** sim=0.8706 dur=6.0 flags=[] judge={"text_fidelity": 4, "endings": 6, "naturalness": 9, "prosody": 7, "accent": 10, "palatalization": 1
  - heard: надо записать щ наверное голос будет хотя
- **cap_enhancement** sim=0.8883 dur=6.0 flags=[] judge={"text_fidelity": 2, "endings": 4, "naturalness": 9, "prosody": 6, "accent": 10, "palatalization": 1
  - heard: надо записать щ наверное голос будет хотя
- **cap_quality** sim=0.8857 dur=6.0 flags=[] judge={"text_fidelity": 4, "endings": 6, "naturalness": 10, "prosody": 8, "accent": 10, "palatalization": 
  - heard: данадо записать еще наверное голос будет хотя
- **cap_speed_12** sim=0.7234 dur=6.0 flags=[] judge={"text_fidelity": 0, "endings": 8, "naturalness": 8, "prosody": 8, "accent": 10, "palatalization": 1
  - heard: надо записать кстать наверное голос голос будет нет нет все нормально я хочу просто записать голос
- **cap_pitch_up** sim=0.4531 dur=6.0 flags=[] judge={"text_fidelity": 0, "endings": 6, "naturalness": 8, "prosody": 6, "accent": 10, "palatalization": 1
  - heard: надо записать наверное голос будет
- **cap_nonverbal_breath** sim=0.8758 dur=6.0 flags=[] judge={"text_fidelity": 10, "endings": 10, "naturalness": 10, "prosody": 10, "accent": 10, "palatalization
  - heard: данадо записать наверное нет нет все нормально я хочу просто записать голос
- **cap_vocal_extraction** sim=0.8837 dur=6.0 flags=[] judge={"text_fidelity": 0, "endings": 4, "naturalness": 8, "prosody": 4, "accent": 10, "palatalization": 1
  - heard: надо записать щ наверное голос будет хотя

## Оговорки
- ASR слеп к ударению/мягкости части фонем — окончательный вердикт за прослушиванием.
- Клиппинг: u0-прогон без limit_peak; продуктовая цепочка его сглаживает.
- Инструменты: корректность операции проверяется отдельно (аудио-эффекты), здесь — сохранение содержания.