"""Сборка фиксированного оценочного пака для u0-контроля и A/B.

Состав:
- TTS: 48 dev-текстов × 4 голоса ротацией (seed 1234, NFE 64)
- Клонирование: 25 val-пар (реф + целевой текст)
- Инструменты: 5 примеров на операцию × 7 (val из tools_v3)
- Каталог: 12 функций из capability_probe (эмоции, шёпот, очистка и т.д.)

Инструкции предвычислены (RuAccent стресс, БЕЗ лексикона — как в обучающих данных).
Запуск пака: run_eval_pack.py (trim=False, bestofn=1 — протокол A/B).
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))

OUT = os.path.join(AUK, "local_tests", "eval_pack")


def main():
    from auk.infer.infer_gradio import _accentize_ru
    from dev_set import TEXTS, VOICES

    os.makedirs(OUT, exist_ok=True)
    pack = []

    def stress(t):
        try:
            return _accentize_ru(t, use_lexicon=False)
        except Exception:
            return t

    # 1) TTS
    voice_names = list(VOICES.keys())
    for i, (cat, text) in enumerate(TEXTS):
        vname = voice_names[i % len(voice_names)]
        ref = VOICES[vname]
        body = stress(text)
        instr = f"Say the following in Russian with clear, natural pronunciation: '{body}'"
        secs = max(3.5, min(12.0, 1.0 + 0.075 * len(text)))
        pack.append({"kind": "tts", "id": f"tts{i:02d}_{cat}_{vname}", "instruction": instr,
                     "ref": ref, "gen_seconds": round(secs, 2), "text": text})

    # 2) Клонирование (val-пары)
    vf = os.path.join(LOCAL, "data_s2_full", "v2_after_identity", "val.jsonl")
    if not os.path.exists(vf):
        vf = os.path.join(LOCAL, "data_s2_full", "val.jsonl")
    n_clone = 0
    with open(vf, encoding="utf-8") as f:
        for line in f:
            if n_clone >= 25:
                break
            o = json.loads(line)
            if len(o["messages"][0]["content"]) < 2:
                continue
            instr = o["messages"][0]["content"][0]["text"]
            ref = o["messages"][0]["content"][1]["audio"]
            secs = float(o.get("duration") or 4.0)
            pack.append({"kind": "clone", "id": f"clone{n_clone:02d}", "instruction": instr,
                         "ref": ref, "gen_seconds": round(secs, 2)})
            n_clone += 1

    # 3) Инструменты (5 на операцию из val)
    tf = os.path.join(LOCAL, "data_s2_tools_v3", "val.jsonl")
    per_op = defaultdict = {}
    n_tools = {}
    with open(tf, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            op = o["meta"]["op"]
            if n_tools.get(op, 0) >= 5:
                continue
            instr = o["messages"][0]["content"][0]["text"]
            in_audio = o["messages"][0]["content"][1]["audio"]
            secs = float(o.get("duration") or 4.0)
            pack.append({"kind": "tool", "id": f"tool_{op}_{n_tools.get(op, 0)}", "instruction": instr,
                         "ref": in_audio, "gen_seconds": round(secs, 2), "op": op})
            n_tools[op] = n_tools.get(op, 0) + 1

    # 4) Каталог (capability_probe список)
    from capability_probe import CASES as CAP_CASES
    cap_ref = os.path.join(AUK, "local_tests", "user_ref.wav")
    for key, instr in CAP_CASES:
        pack.append({"kind": "capability", "id": f"cap_{key}", "instruction": instr,
                     "ref": cap_ref, "gen_seconds": 6.0, "cap": key})

    json.dump(pack, open(os.path.join(OUT, "pack.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter(p["kind"] for p in pack)
    print(f"pack entries: {len(pack)} | {dict(c)}")
    print(f"saved: {os.path.join(OUT, 'pack.json')}")
    print("PACK_BUILD_DONE")


if __name__ == "__main__":
    main()
