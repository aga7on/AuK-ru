"""Тест-2: стресс клонинга на пользовательской базе голосов G:\\AI\\kyutai-ru\\voice.
20+ голосов (муж/жен по возможности), каждая: 1 короткая фраза, 1 длинный текст.
Покрывает: mp3/не-24k, длинные референсы, игровые/кино-голоса, шёпотные/певческие.
Выход: local_tests/user_voices/results.json
"""
import json
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio

AUK = r"G:\AI\AuK"
SRC = r"G:\AI\kyutai-ru\voice"
OUT = os.path.join(AUK, "local_tests", "user_voices")
CKPT = os.path.join(AUK, "local_train", "run_s5", "merged", "auk_s5_4500.safetensors")

SHORT = "привет, это тест клонирования голоса."
LONG_TEXT = ("В далёком лесу жил старый охотник по имени Матвей. "
             "Каждое утро он выходил на тропу, проверял силки и слушал, как просыпается лес. "
             "Однажды зимой он нашёл под ёлкой раненого волка и выходил его целый месяц. "
             "С тех пор волк ходил за ним как собака, и соседи перестали сомневаться, что у Матвея особенный дар.")


def pick_refs():
    files = []
    for f in sorted(os.listdir(SRC)):
        p = os.path.join(SRC, f)
        if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac"):
            try:
                i = sf.info(p)
                files.append((f, p, i.duration))
            except Exception:
                pass
    files = [x for x in files if 4 <= x[2] <= 300]
    # разнообразие: короткие, длинные, разные форматы; явные женские по имени файла где видно
    chosen = []
    fem_kw = ("burceva", "deviant", "female", "deva", "anna", "masha", "olga", "lena")
    for f, p, d in files:
        low = f.lower()
        if any(k in low for k in fem_kw):
            chosen.append((f, p, d, "female?"))
    male_kw = ("andrei", "mine voice", "budkov", "deckard", "geralt", "vaas", "sheriff",
               "criminal", "gofman", "sanych", "trifelev", "ilm")
    for f, p, d in files:
        low = f.lower()
        if any(k in low for k in male_kw) and len(chosen) < 40:
            chosen.append((f, p, d, "male?"))
    # добираем прочими до 24
    for f, p, d in files:
        if (f, p, d, "male?") not in chosen and (f, p, d, "female?") not in chosen:
            chosen.append((f, p, d, "unknown"))
        if len(chosen) >= 24:
            break
    return chosen[:24]


def main():
    os.makedirs(OUT, exist_ok=True)
    refs = pick_refs()
    print(f"refs: {len(refs)}", flush=True)
    for r in refs:
        print("  ", r[0], round(r[2], 1), "s", r[3], flush=True)
    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s5", "merged", "config.yaml"),
                   ckpt_path=CKPT, qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device="cuda:1", dtype="bf16")
    results = []
    t0 = time.time()
    for f, p, d, gender in refs:
        stem = os.path.splitext(f)[0].replace(" ", "_")[:30]
        for label, text in (("short", SHORT), ("long", LONG_TEXT)):
            instr = f"Reproduce the reference voice and say in Russian: '{text}'"
            # для длинного текста — длительность по темпу ~14 сл/с
            secs = min(30.0, (len(text) / 14.0) + 1.0) if label == "long" else (d + 0.5)
            out = os.path.join(OUT, f"{stem}_{label}.wav")
            content = [{"type": "text", "text": instr}, {"type": "audio", "audio": p}]
            try:
                audio, sr = eng.generate([{"role": "user", "content": content}], audio=p,
                                         gen_seconds=float(secs), nfe=64, cfg_strength=2.0, seed=7)
                from auk.infer.quality import normalize_rms, limit_peak
                audio = limit_peak(normalize_rms(audio))
                save_audio(audio, sr, out)
                results.append({"id": f"{stem}_{label}", "source_file": f, "gender_guess": gender,
                                "ref": p, "ref_dur": d, "text": text, "file": out, "status": "ok"})
            except Exception as ex:
                results.append({"id": f"{stem}_{label}", "source_file": f, "gender_guess": gender,
                                "ref": p, "ref_dur": d, "text": text, "file": None,
                                "status": f"error {type(ex).__name__}: {ex}"[:200]})
            print(f"[{len(results)}] {results[-1]['id']} {results[-1]['status'][:60]} ({(time.time()-t0)/60:.1f}m)", flush=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"USER_VOICES_DONE ok={ok}/{len(results)}")


if __name__ == "__main__":
    main()
