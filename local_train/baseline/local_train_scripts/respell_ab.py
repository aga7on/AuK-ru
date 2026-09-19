"""Respell A/B: try hyphen/letter variants for systematically broken words (product path, GigaAM screen)."""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

REF = r"G:\AI\AuK\local_tests\user_ref.wav"

# (phrase, [(variant_name, find, replace), ...]) — replacements applied to the STRESSED text
CASES = [
    ("Электронное объявление появилось на сайте вчера.",
     [("v0_default", None, None),
      ("v1_ob-javlenie", "объявл+ение", "объ-явл+ение"),
      ("v2_obja-vlenie", "объявл+ение", "объяв-л+ение"),
      ("v3_ob-javlenie_soft", "объявл+ение", "обь-явл+ение"),
      ("v4_obajavlenie_short", "объявл+ение", "объявл+енье")]),
    ("Пожалуйста, не опаздывайте на встречу.",
     [("v0_default", None, None),
      ("v1_v-strechu", "встр+ечу", "в-стр+ечу"),
      ("v2_fstrechu", "встр+ечу", "фстр+ечу"),
      ("v3_vs-trechu", "встр+ечу", "вс-тр+ечу")]),
    ("Привет! Это проверка русского произношения. Раз, два, три.",
     [("v0_default", None, None),
      ("v1_short_ya", "произнош+ения", "произнош+енья"),
      ("v2_proiz-noshenia", "произнош+ения", "произ-нош+ения"),
      ("v3_pro-iznoshenia", "произнош+ения", "про-изнош+ения")]),
    ("Администратор объяснил правила пользования сервисом.",
     [("v0_default", None, None),
      ("v1_ad-ministrator", "Администр+атор", "Ад-министр+атор"),
      ("v2_admi-nistrator", "Администр+атор", "Адми-нистр+атор"),
      ("v3_polz-ovania", "п+ользования", "п+ольз-ования"),
      ("v4_po-lzovania", "п+ользования", "п+о-льзования")]),
    ("Люблю грозу в начале мая, когда весенний первый гром.",
     [("v0_default", None, None),
      ("v1_vezen-nii", "вес+енний", "вес+ен-ний"),
      ("v2_vesenn-ii", "вес+енний", "вес+енн-ий"),
      ("v3_ve-sennii", "вес+енний", "ве-с+енний")]),
    ("Сложные времена требуют простых решений.",
     [("v0_default", None, None),
      ("v1_reshen-ii", "реш+ений", "реш+ен-ий"),
      ("v2_re-shenii", "реш+ений", "ре-ш+ений")]),
    ("Мне нравится смотреть на звёзды летними ночами.",
     [("v0_default", None, None),
      ("v1_smotre-t", "смотр+еть", "смотр+е-ть"),
      ("v2_s-motret", "смотр+еть", "с-мотр+еть"),
      ("v3_smot-ret", "смотр+еть", "смот-р+еть")]),
]


def estimate_seconds(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
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
    for pi, (text, variants) in enumerate(CASES):
        stressed = _accentize_ru(text)
        secs = estimate_seconds(text)
        for name, find, repl in variants:
            body = stressed if find is None else stressed.replace(find, repl)
            assert find is None or body != stressed, f"replace failed: {find} in {stressed}"
            instr = f"Say the following with the same voice: '{body}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "audio", "audio": REF},
            ]}]
            audio, sr = engine.generate(messages, audio=REF, gen_seconds=secs, nfe=64,
                                        cfg_strength=2.0, seed=args.seed)
            out = os.path.join(args.out, f"rp{pi}_r{len(manifest)}.{_safe(name)}.wav")
            save_audio(audio, sr, out)
            manifest.append({"phrase": pi, "variant": name, "text": text, "body": body,
                             "gen": out, "ref": REF, "gen_seconds": secs, "seed": args.seed})
            print(f"generated {out}", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("RESPELL_AB_DONE", flush=True)


def _safe(s):
    return "".join(ch if ch.isalnum() else "-" for ch in s)


if __name__ == "__main__":
    main()
