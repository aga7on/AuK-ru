# -*- coding: utf-8 -*-
"""S17 probe 1: морфемная разметка сверхдлинных слов (дефисы) vs как есть.

Гипотеза (S17_FRONTEND_V2_DESIGN): вставка дефисов между морфемами помогает модели
не разваливать слово (v1.0: «рентгеноэлектрокардиографический» → «рентгеновая
электрокардиографичестика», CER 0.154).

Метод: 4 длинных слова × {plain, hyphen} × 2 сида = 16 генераций.
Метрика: CER против ожидаемого (нормализованного, дефисы сняты) — wer_norm.

usage: python s17_longword_probe.py [--device cuda:1]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "s17_longword_probe")

CASES = [
    ("рентгеноэлектрокардиографический аппарат",
     "рентгено-электро-кардио-графический аппарат"),
    ("частнопредпринимательская деятельность",
     "частно-предпринимательская деятельность"),
    ("административно-территориальное деление",
     "административно-территориальное деление"),
    ("превосходительство распорядился о субстантивировании",
     "превосходительство распорядился о суб-стан-ти-ви-ро-ва-нии"),
    ("четырехсотдвадцатипятилетний юбилей",
     "четырехсот-двадцати-пяти-летний юбилей"),
    ("сельскохозяйственное машиностроение",
     "сельско-хозяйственное машино-строение"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from auk.infer.ru_frontend import to_speakable
    from ru_metrics import transcribe_path, text_metrics
    from wer_norm import norm_for_wer

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    ref = next(e["ref"] for e in pack if e.get("kind") == "clone" and e["id"] == "clone08")

    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"),
                   ckpt_path=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    rows = []
    for ci, (plain, hyph) in enumerate(CASES):
        expected = norm_for_wer(plain.replace("-", " "))
        for form, text in (("plain", plain), ("hyphen", hyph)):
            speak = to_speakable(text)
            instr = f"Say the following in Russian with clear, natural pronunciation: '{speak}'"
            for seed in (7, 999):
                fp = os.path.join(OUT, f"c{ci}_{form}_s{seed}.wav")
                audio, sr = eng.generate([{"role": "user", "content": [
                    {"type": "text", "text": instr}]}], audio=ref,
                    gen_seconds=7.0, nfe=64, cfg_strength=2.0, seed=seed)
                audio = limit_peak(normalize_rms(audio))
                save_audio(audio, sr, fp)
                heard, err = transcribe_path(fp)
                tm = text_metrics(expected, norm_for_wer(heard))
                rows.append({"case": ci, "form": form, "seed": seed, "plain": plain,
                             "fed": speak, "wer_norm": round(float(tm["wer"]), 3),
                             "cer_norm": round(float(tm.get("cer", 0)), 3),
                             "heard": (heard or "")[:110]})
                print(f"c{ci} {form} s{seed} wer={rows[-1]['wer_norm']:.2f} "
                      f"heard={(heard or '')[:55]}", flush=True)

    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    import statistics as st
    print("--- by form ---")
    for form in ("plain", "hyphen"):
        rs = [r for r in rows if r["form"] == form]
        print(f"{form}: n={len(rs)} wer_norm med {st.median(r['wer_norm'] for r in rs):.3f} "
              f"mean {st.mean(r['wer_norm'] for r in rs):.3f} "
              f"cer med {st.median(r['cer_norm'] for r in rs):.3f}")
    # попарно
    wins = 0
    for ci in range(len(CASES)):
        p = st.median([r["wer_norm"] for r in rows if r["case"] == ci and r["form"] == "plain"])
        h = st.median([r["wer_norm"] for r in rows if r["case"] == ci and r["form"] == "hyphen"])
        print(f"case {ci}: plain {p:.2f} vs hyphen {h:.2f} -> {'HYPHEN' if h < p else ('PLAIN' if p < h else 'TIE')}")
        if h < p:
            wins += 1
    print(f"LONGWORD_PROBE hyphen wins {wins}/{len(CASES)}")


if __name__ == "__main__":
    main()
