"""Dev-набор (шаг 3): 48 новых текстов, 4 реальных голоса, seed 1234, БЕЗ trim и best-of
(«чистый» режим замера). Генерация на cuda:0 (UI пользователя живёт на cuda:1).
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
OUT = os.path.join(AUK, "local_tests", "dev_set", "u18000")

VOICES = {
    "user_male": os.path.join(AUK, "local_tests", "user_ref.wav"),
    "nat_female": os.path.join(AUK, "local_tests", "ref_ru.wav"),
    "alt_male": os.path.join(AUK, "local_tests", "ref_male2.wav"),
    "alt_female": os.path.join(AUK, "local_tests", "ref_female2.wav"),
}

TEXTS = [
    ("clusters", "Вздрогнул от неожиданности и вскрикнул."),
    ("clusters", "Съёмка фильма начнётся в конце месяца."),
    ("clusters", "Объёмный звук наполнил комнату."),
    ("clusters", "Всплеск эмоций был неожиданным."),
    ("clusters", "Столкновение интересов привело к спорам."),
    ("clusters", "Впрыск топлива происходит автоматически."),
    ("soft_sign", "Пекарь замесил тесто с самого утра."),
    ("soft_sign", "Февраль выдался на удивление тёплым."),
    ("soft_sign", "Борьба за качество продолжается."),
    ("soft_sign", "Скользкий лёд покрыл тротуары."),
    ("soft_sign", "Сельский пейзаж радовал глаз."),
    ("soft_sign", "Возьми фонарь и спички."),
    ("reduction", "Сейчас приду, подожди немного."),
    ("reduction", "Вообще-то я уже всё сделал."),
    ("reduction", "Кажется, что дождь скоро закончится."),
    ("reduction", "Купи что-нибудь к чаю."),
    ("reduction", "Где-нибудь здесь должен быть выход."),
    ("reduction", "Когда-то здесь был старый парк."),
    ("numerals", "Сорок семь участников зарегистрировались заранее."),
    ("numerals", "Девятьсот двадцать рублей — точная сумма."),
    ("numerals", "Пятьдесят шесть процентов проголосовали."),
    ("numerals", "К две тысячи двадцать пятому году всё изменится."),
    ("numerals", "Три четверти пути уже позади."),
    ("numerals", "Сто восемьдесят три страницы — это много."),
    ("names", "Екатерина подготовила отчёт для совета."),
    ("names", "Дмитрий встретил нас у входа."),
    ("names", "Наталья Семёновна ведёт литературный кружок."),
    ("names", "Александр Ильич любит шахматы."),
    ("names", "Любовь Петровна испекла пирог."),
    ("names", "Константин работает над новым проектом."),
    ("prosody", "Когда солнце село, на улице стало прохладно, и мы вернулись домой."),
    ("prosody", "Он открыл окно, вдохнул свежий воздух и улыбнулся."),
    ("prosody", "Сначала мы обсудили план, затем распределили задачи, и работа закипела."),
    ("prosody", "Если ты закончишь раньше, позвони мне, пожалуйста."),
    ("prosody", "Дождь закончился так же внезапно, как и начался."),
    ("prosody", "Птицы вернулись с юга, а вместе с ними пришла весна."),
    ("baseline", "Сегодня мы обсудим несколько важных вопросов."),
    ("baseline", "Пожалуйста, оставьте свои замечания в конце документа."),
    ("baseline", "Завтра утром я отправлю вам подробный ответ."),
    ("baseline", "Откройте вторую страницу и прочитайте первый абзац."),
    ("baseline", "Спасибо всем за участие в сегодняшней встрече."),
    ("baseline", "Мы продолжим работу над проектом на следующей неделе."),
    ("hard_words", "Любопытство взяло верх над осторожностью."),
    ("hard_words", "Впечатление от поездки осталось незабываемым."),
    ("hard_words", "Велосипед стоял у самого подъезда."),
    ("hard_words", "Благодарность выразили всем организаторам."),
    ("hard_words", "Здравствуйте, рады видеть вас снова."),
    ("hard_words", "Радость встречи переполняла нас."),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
        ckpt_path=MERGED,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:0",   # UI живёт на cuda:1 — не мешаем пользователю
        dtype="bf16",
    )
    voice_names = list(VOICES.keys())
    manifest = []
    for i, (cat, text) in enumerate(TEXTS):
        vname = voice_names[i % len(voice_names)]
        ref = VOICES[vname]
        body = _accentize_ru(text)
        instr = f"Say the following in Russian with clear, natural pronunciation: '{body}'"
        secs = estimate_seconds(text)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=1234)
        out = os.path.join(OUT, f"dev{i:02d}_{cat}_{vname}.wav")
        save_audio(audio, sr, out)   # RAW: без trim
        manifest.append({"idx": i, "cat": cat, "text": text, "voice": vname, "ref": ref,
                         "file": out, "gen_seconds": secs, "seed": 1234})
        print(f"dev{i:02d} {cat:<10} {vname:<11} | {text[:48]}", flush=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("DEV_SET_DONE")


if __name__ == "__main__":
    main()
