"""Фонетический пак: перенос ж/з, ч/ц, ы/и на новых словах (не только одна фраза).

8 фраз × 2 голоса (user_male, nat_female) = 16 заданий. Формат совместим с run_eval_pack.py
(trim=False, bestofn=1). Инструкции: RuAccent (use_lexicon=False) — как в обучающих данных.

usage: phonetic_pack_build.py [--out G:\AI\AuK\local_tests\phonetic_pack]
"""
import argparse
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

PHRASES = [
    "Жёлтый жук жужжит над розами уже два часа.",
    "Через час чёрный чемодан починят в мастерской.",
    "Цапля цокала у цементного крыльца.",
    "Милые мыши были сыты, вы их не будите.",
    "Женя ждёт чужую машину у жилого дома.",
    "Цыплёнок чутко чирикал целый час.",
    "Зебра зажмурилась от жёлтого заката.",
    "Уже через час шесть жёлтых стрижей улетят.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"G:\AI\AuK\local_tests\phonetic_pack")
    args = ap.parse_args()

    from auk.infer.infer_gradio import _accentize_ru
    from dev_set import VOICES

    os.makedirs(args.out, exist_ok=True)
    voices = {"user_male": VOICES["user_male"], "nat_female": VOICES["nat_female"]}

    def stress(t):
        try:
            return _accentize_ru(t, use_lexicon=False)
        except Exception:
            return t

    pack = []
    i = 0
    for phrase in PHRASES:
        body = stress(phrase)
        instr = f"Say the following in Russian with clear, natural pronunciation: '{body}'"
        secs = max(3.5, min(12.0, 1.0 + 0.075 * len(phrase)))
        for vname, ref in voices.items():
            pack.append({"kind": "phonetics", "id": f"ph{i:02d}_{vname}", "instruction": instr,
                         "ref": ref, "gen_seconds": round(secs, 2), "text": phrase})
            i += 1
    json.dump(pack, open(os.path.join(args.out, "pack.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"phonetic pack: {len(pack)} entries -> {args.out}\\pack.json")
    for p in pack:
        print(" ", p["id"], "|", p["text"])
    print("PHON_PACK_DONE")


if __name__ == "__main__":
    main()
