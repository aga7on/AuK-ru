"""Догон user_voices: упавшие OOM-рефы (>=39с) — генерация с усечённым референсом 30с.
Проверка гипотезы: длинный реф → OOM; обрезка рефа решает. Это и есть обход для пользователей.
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
OUT = os.path.join(AUK, "local_tests", "user_voices")
CKPT = os.path.join(AUK, "local_train", "run_s5", "merged", "auk_s5_4500.safetensors")
TRIM_DIR = os.path.join(OUT, "_refs_30s")
MAX_REF_S = 30.0

SHORT = "привет, это тест клонирования голоса."
LONG_TEXT = ("В далёком лесу жил старый охотник по имени Матвей. "
             "Каждое утро он выходил на тропу, проверял силки и слушал, как просыпается лес. "
             "Однажды зимой он нашёл под ёлкой раненого волка и выходил его целый месяц. "
             "С тех пор волк ходил за ним как собака, и соседи перестали сомневаться, что у Матвея особенный дар.")


def trim_ref(src):
    os.makedirs(TRIM_DIR, exist_ok=True)
    dst = os.path.join(TRIM_DIR, os.path.splitext(os.path.basename(src))[0][:30] + "_30s.wav")
    if not os.path.exists(dst):
        a, sr = sf.read(src)
        if len(a) / sr > MAX_REF_S:
            a = a[: int(MAX_REF_S * sr)]
        sf.write(dst, a, sr)
    return dst


def main():
    results = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    failed = [r for r in results if r["status"] != "ok"]
    print("failed:", len(failed), flush=True)
    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s5", "merged", "config.yaml"),
                   ckpt_path=CKPT, qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device="cuda:1", dtype="bf16")
    from auk.infer.quality import normalize_rms, limit_peak
    t0 = time.time()
    for r in failed:
        ref = trim_ref(r["ref"])
        label = "short" if r["id"].endswith("_short") else "long"
        text = r["text"]
        secs = min(30.0, (len(text) / 14.0) + 1.0) if label == "long" else MAX_REF_S + 0.5
        instr = f"Reproduce the reference voice and say in Russian: '{text}'"
        content = [{"type": "text", "text": instr}, {"type": "audio", "audio": ref}]
        try:
            audio, sr = eng.generate([{"role": "user", "content": content}], audio=ref,
                                     gen_seconds=float(secs), nfe=64, cfg_strength=2.0, seed=7)
            audio = limit_peak(normalize_rms(audio))
            out = os.path.join(OUT, f"{r['id']}_ref30s.wav")
            save_audio(audio, sr, out)
            r.update({"file": out, "status": "ok_ref30s", "ref_used_trimmed": ref})
        except Exception as ex:
            r.update({"status": f"error2 {type(ex).__name__}: {ex}"[:200]})
        print(f"[{r['id']}] {r['status'][:60]} ({(time.time()-t0)/60:.1f}m)", flush=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["status"].startswith("ok"))
    print(f"USER_VOICES_RECOVER done ok={ok}/{len(results)}")


if __name__ == "__main__":
    main()
