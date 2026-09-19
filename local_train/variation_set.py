"""Variation test set: 20 diverse Russian phrases, alternating male/female refs, cyrillic+stress."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

MALE_REF = os.path.join(AUK, "local_tests", "user_ref.wav")
FEMALE_REF = os.path.join(AUK, "local_tests", "ref_ru.wav")

PHRASES = [
    ("Иван Петров работает в банке с две тысячи двадцать шестого года.", 7.5),
    ("Договор подписан, каталог обновлён, а он всё ещё звонит.", 7.0),
    ("Позвони мне, пожалуйста, когда освободишься.", 6.0),
    ("Съешь ещё этих мягких французских булок, да выпей чаю.", 7.0),
    ("Вся семья собралась за столом, и объём работы уже не пугал.", 7.5),
    ("Шла Саша по шоссе и сосала сушку.", 5.5),
    ("Ты придёшь завтра на встречу?", 5.0),
    ("Как же здорово, что мы вместе!", 5.0),
    ("Интернет-магазин открылся в новом торговом центре.", 6.5),
    ("Достопримечательности Санкт-Петербурга впечатляют туристов.", 7.5),
    ("Тридцать три корабля лавировали, лавировали, да не вылавировали.", 8.0),
    ("Рок-н-ролл и джаз звучали из открытого окна.", 6.5),
    ("Пожалуйста, включи свет в коридоре и закрой дверь.", 6.5),
    ("Мы обсудили стратегию на следующий квартал и бюджет.", 7.0),
    ("Взгляд её был спокоен, а голос звучал уверенно.", 6.5),
    ("Файл сохранён в папке «Документы» на рабочем столе.", 7.0),
    ("На юге России растёт виноград и спеют персики.", 6.5),
    ("Он объяснил, почему проект задержался на две недели.", 7.0),
    ("Четыре чёрненьких чумазеньких чертёнка чертили чёрными чернилами чертёж.", 9.0),
    ("Электричка отправляется через пятнадцать минут с третьего пути.", 7.5),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_5000.safetensors"))
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "variations"))
    ap.add_argument("--nfe", type=int, default=32)
    args = ap.parse_args()
    CKPT = args.ckpt
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(CKPT), "config.yaml"),
        ckpt_path=CKPT,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )
    manifest = []

    def gen_item(i, text, ref, secs, tag=None, seed=42):
        body = _accentize_ru(text)
        instr = f"Say the following with the same voice: '{body}'"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=args.nfe,
                                    cfg_strength=2.0, seed=seed)
        name = f"var_{i:02d}{('_' + tag) if tag else ''}_s{seed}_{'m' if i % 2 == 0 else 'f'}.wav"
        out = os.path.join(OUT, name)
        save_audio(audio, sr, out)
        manifest.append({"idx": i, "text": text, "ref": ref, "gen": out, "gen_seconds": secs,
                         "representation": "cyr_stress", "seed": seed,
                         "voice": "male" if i % 2 == 0 else "female", "tag": tag or ""})
        print(f"generated {out} ({secs:.1f}s)", flush=True)

    for i, (text, _old) in enumerate(PHRASES):
        ref = MALE_REF if i % 2 == 0 else FEMALE_REF
        secs = estimate_seconds(text)
        gen_item(i, text, ref, secs, seed=1234)
        gen_item(i, text, ref, secs, seed=7)
        if i == 14:
            gen_item(i, text, ref, max(3.5, secs - 1.5), tag="short", seed=1234)
            gen_item(i, text, ref, min(12.0, secs + 1.5), tag="long", seed=1234)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("VARIATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
