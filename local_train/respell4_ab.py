"""Respell round 4: fix ъ-swallowing (объяснял/объём) and 'энергии' over-emphasis.

User verdict on r3p0_v3: 'об(ъ) знак проглочен' (объяснял), 'энергии' -> надo 'энэргии'
(schwa-like [э], не выделенное [е]).
"""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = os.path.join(AUK, "local_tests", "user_ref.wav")

CASES = [
    # p0: энергии -> энэргии (schwa-like э, no dash); объяснял ъ-fix; учитель/сохранения from r3p0 v3 baseline
    ("Учитель физики объяснял законы сохранения энергии.",
     [("v0_default", []),
      ("v1_energii_schwa", [("эн+ергии", "эн+эргии")]),
      ("v2_so-hran_energii_schwa", [("сохран+ения", "со-хран+ения"), ("эн+ергии", "эн+эргии")]),
      ("v3_uchitel_sohran_schwa", [("Уч+итель", "Учи-т+ель"), ("сохран+ения", "сох-ран+ения"), ("эн+ергии", "эн+эргии")]),
     ]),
    # p1: ъ-swallowing in объём/съели (user: 'р3p1_v2_phonetic_full — проглатывание')
    ("Объявление: объём работы большой, но пять друзей съели весь борщ.",
     [("v0_default", []),
      ("v1_obyom_ee", [("объём", "объ+ём"), ("съ+ели", "съ+йели")]),
      ("v2_obyom_o_soft", [("объём", "обь+ём"), ("съ+ели", "съ-йели")]),
     ]),
    # p2: verify c-мо-тр+еть (LEXICON now applies automatically!) + ъ in объяснял analog
    ("Мне нравится смотреть на звёзды летними ночами.",
     [("v0_default", []),
      ("v1_ux_smo-tret", [("смотр+еть", "с-мо-тр+еть")]),
     ]),
    # p3: пользования сервисом — стык глотается ('ползавония'); test hyphen + explicit break
    ("Администратор объяснил правила пользования сервисом.",
     [("v0_default", []),
      ("v1_polz_saervisom_break", [("п+ольз-ования с+ервисом", "п+ольз-ования. с+ервисом")]),
      ("v2_polz_ee", [("п+ольз-ования", "п+ольз-авания")]),
      ("v3_polz_o_full", [("п+ольз-ования", "по-льзо-вания")]),
     ]),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.08 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors"))
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "respell4_ab"))
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
                if find not in body:
                    print(f"WARN: '{find}' not in body, skipping for {name}")
                    continue
                body = body.replace(find, repl)
            instr = f"Say the following with the same voice: '{body}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=args.seed)
            audio_trimmed = quality.trim_silence(audio, sr)
            out = os.path.join(args.out, f"r4p{pi}_{name}.wav")
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
            print(f"p{pi} {name:<28} | recall={rec:.2f} | {heard[:55]}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("RESPELL4_AB_DONE")


if __name__ == "__main__":
    main()
