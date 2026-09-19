"""Respell round 3: tackle ap19 (Учитель/сохранения), ap10 (объём/большой/съели),

смотреть (т-drop), весенний/решений (voicing), Администратор.
Run on auk_ru_18000.safetensors with seed 1234.
"""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")

CASES = [
    # Phrase 0: ap19 (Учитель, сохранения, энергии)
    ("Учитель физики объяснял законы сохранения энергии.",
     [("v0_default", []),
      ("v1_uchi-tel_soh-ranenia", [("Уч+итель", "Учи-т+ель"), ("сохран+ения", "сох-ран+ения")]),
      ("v2_uchi-tel_so-hranenia", [("Уч+итель", "Уч+и-тель"), ("сохран+ения", "со-хран+ения")]),
      ("v3_sah-ranenia_e-nergii", [("Уч+итель", "Уч+итель."), ("сохран+ения", "сах-ран+ения"), ("эн+ергии", "э-н+ергии")]),
     ]),
    # Phrase 1: ap10 (объём, большой, съели)
    ("Объявление: объём работы большой, но пять друзей съели весь борщ.",
     [("v0_default", []),
      ("v1_hyphen_clusters", [("объём", "об-ъ+ём"), ("больш+ой", "боль-ш+ой"), ("съ+ели", "съ-й+ели")]),
      ("v2_phonetic_full", [("объём", "аб-й+ом"), ("больш+ой", "баль-ш+ой"), ("съ+ели", "с-й+ели")]),
      ("v3_soft_clusters", [("объём", "обь-й+ом"), ("больш+ой", "бол-ьш+ой"), ("съ+ели", "съ-ели")]),
     ]),
    # Phrase 2: ap03 (смотреть -> смареть)
    ("Мне нравится смотреть на звёзды летними ночами.",
     [("v0_default", []),
      ("v1_smottret_geminate", [("смотр+еть", "смоттр+еть")]),
      ("v2_smot-tret", [("смотр+еть", "смот-т+реть")]),
      ("v3_smat-tret_phonetic", [("смотр+еть", "смат-т+реть")]),
      ("v4_s-mo-tret", [("смотр+еть", "с-мо-тр+еть")]),
     ]),
    # Phrase 3: ap21 (весенний -> визений)
    ("Люблю грозу в начале мая, когда весенний первый гром.",
     [("v0_default", []),
      ("v1_ves-sennii", [("вес+енний", "вес-с+енний")]),
      ("v2_ve-s-sennii", [("вес+енний", "ве-с-с+енний")]),
     ]),
    # Phrase 4: ap26 (решений -> режений)
    ("Сложные времена требуют простых решений.",
     [("v0_default", []),
      ("v1_resh-shenii", [("реш+ений", "реш-ш+ений")]),
      ("v2_re-sh-shenii", [("реш+ений", "ре-ш-ш+ений")]),
     ]),
    # Phrase 5: ap25 (Администратор -> Адмнистратор)
    ("Администратор объяснил правила пользования сервисом.",
     [("v0_default", []),
      ("v1_ad-mi-nistrator", [("Администр+атор", "Ад-ми-нистр+атор")]),
      ("v2_admin-istrator", [("Администр+атор", "Админ-истр+атор")]),
     ]),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.08 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors"))
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "respell3_ab"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.infer_gradio import _accentize_ru
    from auk.infer import quality

    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
        ckpt_path=args.ckpt,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:1",
        dtype="bf16",
    )

    manifest = []
    for pi, (text, variants) in enumerate(CASES):
        stressed = _accentize_ru(text)
        secs = estimate_seconds(text)
        for name, replacements in variants:
            body = stressed
            for find, repl in replacements:
                assert find in body, f"Replace failed: {find} not in {body}"
                body = body.replace(find, repl)
            instr = f"Say the following with the same voice: '{body}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=args.seed)
            audio_trimmed = quality.trim_silence(audio, sr)
            out = os.path.join(args.out, f"r3p{pi}_{name}.wav")
            save_audio(audio_trimmed, sr, out)
            
            heard = quality.transcribe(audio_trimmed, sr)
            rec = quality.recall(text, heard)
            
            manifest.append({
                "phrase_idx": pi,
                "variant": name,
                "text": text,
                "body": body,
                "file": out,
                "recall": round(rec, 3),
                "asr_heard": heard,
            })
            print(f"p{pi} {name:<24} | recall={rec:.2f} | {heard[:55]}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("\nRESPELL3_AB_DONE")


if __name__ == "__main__":
    main()
