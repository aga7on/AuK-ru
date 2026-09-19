"""D-mini autopsy: 45 phrases x 2 seeds through the product path (AukInfer, cyr_stress + lexicon)."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

USER_REF = r"G:\AI\AuK\local_tests\user_ref.wav"
FEM_REF = r"G:\AI\AuK\local_tests\ref_ru.wav"

PHRASES = [
    # val / hard words
    ("Ещё более важную роль на Африканском Роге играет устойчивое развитие.", "hard_words"),
    ("Роль книги в жизни человека трудно переоценить.", "hard_words"),
    ("Цель проекта — устойчивое развитие региона.", "hard_words"),
    ("Мне нравится смотреть на звёзды летними ночами.", "hard_words"),
    # reduction / prosody traps
    ("Позвони мне, пожалуйста, когда освободишься.", "reduction"),
    ("Пожалуйста, не опаздывайте на встречу.", "reduction"),
    ("Не забудь позвонить маме, она очень волнуется.", "reduction"),
    # names / prepositions
    ("Иван Петров хранит деньги в банке с двумя тысячами рублями.", "names"),
    ("Ольга и Сергей поехали в деревню к бабушке.", "names"),
    # clusters
    ("Съешь ещё этих мягких французских булок, да выпей чаю.", "clusters"),
    ("Объявление: объём работы большой, но пять друзей съели весь борщ.", "clusters"),
    ("Экскурсовод рассказал о древних объёмных сооружениях.", "clusters"),
    ("Взгляд её был спокоен, и голос звучал ровно.", "clusters"),
    # stress / omographs
    ("Мы проверяем ударения: договор, каталог, звонит, красивее.", "stress"),
    ("Замок стоит на горе, а на двери висит большой замок.", "stress"),
    # numerals as words
    ("Две тысячи двадцать шестого года мы запустили этот проект.", "numerals"),
    ("Тысяча девятьсот восемьдесят пятый год стал переломным.", "numerals"),
    # soft sign / palatalization
    ("Вся семья была дома, и день прошёл отлично.", "soft_sign"),
    ("Сельдь под шубой — традиционное новогоднее блюдо.", "soft_sign"),
    ("Учитель физики объяснял законы сохранения энергии.", "soft_sign"),
    # long fluent prosody
    ("Дождь стучал по крыше, а ветер гнул деревья.", "prosody"),
    ("Люблю грозу в начале мая, когда весенний первый гром.", "prosody"),
    ("Каждый вечер он гулял по набережной и слушал музыку.", "prosody"),
    ("Он смотрел на неё и не мог поверить своим глазам.", "prosody"),
    ("Берегите природу — она наш общий дом.", "prosody"),
    ("Администратор объяснил правила пользования сервисом.", "prosody"),
    ("Сложные времена требуют простых решений.", "prosody"),
    ("Мой дедушка любил рассказывать истории о войне.", "prosody"),
    ("Девушка улыбнулась и тихо сказала: «До встречи».", "prosody"),
    ("Библиотека закрывается ровно в восемь часов вечера.", "prosody"),
    # plain sentences (baseline)
    ("Привет! Это проверка русского произношения. Раз, два, три.", "baseline"),
    ("Сегодня отличный день, и мы тестируем новую систему синтеза.", "baseline"),
    ("Он сказал: «Я вернусь завтра утром», и вышел из комнаты.", "baseline"),
    ("В лесу пахло сыростью и хвоей.", "baseline"),
    ("Тёплый ветер принёс запах моря.", "baseline"),
    ("В меню были пельмени, борщ и компот.", "baseline"),
    ("Компьютер включился быстрее, чем я ожидал.", "baseline"),
    ("Этот фильм произвёл на меня сильное впечатление.", "baseline"),
    ("Инженеры строят новый мост через реку.", "baseline"),
    ("Программисты пишут код, а тестировщики его проверяют.", "baseline"),
    ("Спасибо большое за тёплый приём и поддержку.", "baseline"),
    ("Электронное объявление появилось на сайте вчера.", "baseline"),
    ("С детства он мечтал стать врачом и спасать людей.", "baseline"),
    ("На улице шёл мелкий осенний дождь.", "baseline"),
    ("Всё вокруг казалось огромным и немного странным.", "baseline"),
]

SEEDS = [1234, 7]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []
    for idx, (text, cat) in enumerate(PHRASES):
        ref = FEM_REF if idx % 9 == 4 else USER_REF
        secs = estimate_seconds(text)
        body = _accentize_ru(text)
        instr = f"Say the following with the same voice: '{body}'"
        for seed in SEEDS:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": ref},
            ]}]
            audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=seed)
            out = os.path.join(args.out, f"ap{idx:02d}_{cat}_s{seed}.wav")
            save_audio(audio, sr, out)
            manifest.append({"idx": idx, "text": text, "category": cat, "ref": ref,
                             "gen": out, "gen_seconds": secs, "seed": seed,
                             "representation": "cyr_stress", "instruction": instr})
            print(f"generated {out}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("AUTOPSY_DONE", flush=True)


if __name__ == "__main__":
    main()
