"""Build the preflight manifest (12 real tasks + positive/negative controls) for judge v2."""
import csv
import json
import os

import numpy as np
import soundfile as sf

from audit_gemini_judge_v2 import task_spec

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
BLIND = os.path.join(LT, "blind_s2")
OUTDIR = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
CTRL = os.path.join(OUTDIR, "preflight_controls")
os.makedirs(CTRL, exist_ok=True)

TASKS = [
    ("tts", "tts00_clusters_user_male"),
    ("tts", "tts28_names_user_male"),
    ("phonetics", "ph00_user_male"),
    ("phonetics", "ph02_user_male"),
    ("phrase_training", "phrase_sber"),
    ("clone", "clone00"),
    ("tool", "tool_volume_up_0"),
    ("tool", "tool_noise_add_0"),
    ("tool", "tool_pitch_up_0"),
    ("capability", "cap_emotion_happy"),
    ("capability", "cap_speech_edit"),
    ("capability", "cap_vocal_extraction"),
]


def load_refs():
    pack = json.load(open(os.path.join(LT, "eval_pack", "pack.json"), encoding="utf-8"))
    phon = json.load(open(os.path.join(LT, "phonetic_pack", "pack.json"), encoding="utf-8"))
    return {p["id"]: p for p in list(pack) + list(phon)}


def main():
    phrase = "Сбер, включи музыку для пробежек через десять минут"
    rows = list(csv.DictReader(open(os.path.join(BLIND, "LISTEN.csv"), encoding="utf-8")))
    first = {}
    for r in rows:
        first.setdefault(r["task_id"], r["file"])
    refs = load_refs()
    items = []
    for group, tid in TASKS:
        f = first.get(tid)
        if not f:
            print("SKIP missing task", tid)
            continue
        ref = (os.path.join(LT, "user_ref.wav") if tid == "phrase_sber"
               else refs[tid].get("ref"))
        spec = ({"mode": "tts", "needs_input": True, "text": phrase, "task_name": "phrase_training",
                 "goal": "", "checks": ""} if tid == "phrase_sber"
                else task_spec(refs[tid], group, tid))
        items.append({"file": f, "task_id": tid, "group": group, "ref": ref,
                      "wav": os.path.join(BLIND, "wav", f), "control": "real", **spec})

    src = os.path.join(AUK, "local_train", "run_s2_A", "samples", "update_500_tgt.wav")
    audio, sr = sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    user_ref = os.path.join(LT, "user_ref.wav")

    def ctrl(name, data, text, task_id, control):
        p = os.path.join(CTRL, name)
        sf.write(p, data, sr)
        return {"file": name, "task_id": task_id, "group": "tts", "ref": user_ref,
                "wav": p, "text_override": text, "control": control}

    ph_text = refs["ph00_user_male"]["text"]
    controls = [
        ctrl("pos_target_phrase.wav", audio, phrase, "preflight_pos", "positive_real_target"),
        ctrl("neg_silence.wav", np.zeros_like(audio), phrase, "preflight_neg_silence", "negative_silence"),
        ctrl("neg_trunc.wav", audio[: int(0.4 * sr)], phrase, "preflight_neg_trunc", "negative_truncated"),
        ctrl("neg_noise.wav", audio + 0.15 * np.random.default_rng(0).standard_normal(len(audio)).astype("float32"),
             phrase, "preflight_neg_noise", "negative_noise"),
        ctrl("neg_wrong_text.wav", audio, ph_text, "preflight_neg_wrong", "negative_content_mismatch"),
    ]
    for c in controls:
        c["wav"] = os.path.abspath(c["wav"])
    items += controls

    out = os.path.join(OUTDIR, "preflight_manifest.json")
    json.dump(items, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("manifest", out, "items", len(items))
    for it in items:
        print(" ", it["control"], it["group"], it["task_id"], it["file"])


if __name__ == "__main__":
    main()
