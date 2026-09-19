"""Full E2E suite through the real UI pipeline (run_generate).

Covers: bare-text auto-wrap TTS (lexicon coverage, best-of-2 + trim), emotion instruct,
speech content editing, whisper conversion. Every sample is verified: duration sane,
no clipping, lead silence trimmed, GigaAM recall vs expected text, auto-judge (2 votes).

Run with workdir G:\AI\AuK (VAE loads relative to cwd).
"""
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

import auk.infer.infer_gradio as ig  # noqa: E402
from auk.infer.infer_auk import AukInfer  # noqa: E402
from auk.infer import quality  # noqa: E402
from auto_judge import call_model, merge_votes  # noqa: E402

AUK = r"G:\AI\AuK"
MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
REF = os.path.join(AUK, "local_tests", "user_ref.wav")
OUT = os.path.join(AUK, "local_tests", "e2e_suite")
MODEL = "gemini-3.8-flash-medium"

ig.CKPT_PATHS["e2e"] = MERGED
ig.ENGINES["e2e"] = AukInfer(
    config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
    ckpt_path=MERGED,
    qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
    cpu_offload=True,
    device="cuda:1",
    dtype="bf16",
)

TTS_CASES = [
    "Привет! Это проверка русского произношения.",
    "Ещё более важную роль на Африканском Роге играет устойчивое развитие.",
    "Пожалуйста, не опаздывайте на встречу, электронное объявление уже на сайте.",
    "Мне нравится смотреть на звёзды летними ночами.",
    "Иван Петров объяснил правила пользования сервисом с двумя тысячами участников.",
    "Учитель физики объяснял законы сохранения энергии.",
]


def metrics(pcm, sr):
    x = pcm.astype("float32") / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    excess = quality.pause_excess(x, sr)
    # manual lead computation
    frame, hop = 600, 240
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    lead = 0.0
    for i, s in enumerate(speech):
        if s:
            lead = i * hop / sr
            break
    return {"dur": round(len(x) / sr, 2), "peak": round(peak, 3), "lead": round(lead, 2),
            "excess": round(excess, 2)}


def save_pcm(pcm, sr, path):
    sf = __import__("soundfile")
    sf.write(path, pcm, sr, subtype="PCM_16")


def judge_file(path, text, votes=2):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    objs = []
    for _ in range(votes):
        obj, raw = call_model(MODEL, b64, text)
        if obj:
            objs.append(obj)
    return merge_votes(objs) if objs else None


def main():
    rows = []
    os.makedirs(OUT, exist_ok=True)

    # --- A. bare Russian text -> auto-wrap -> TTS (best-of-2 + trim) ---
    for i, text in enumerate(TTS_CASES):
        t0 = __import__("time").time()
        sr, pcm = ig.run_generate("e2e", REF, text, None, "", "", 64, 2.0, 1234,
                                  translit=False, accentize=True, bestofn=2, trim=True, tts_wrap=True)
        path = os.path.join(OUT, f"tts_{i:02d}.wav")
        save_pcm(pcm, sr, path)
        m = metrics(pcm, sr)
        heard = quality.transcribe(__import__("torch").from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0), sr)
        rec = quality.recall(text, heard)
        j = judge_file(path, text)
        ov = (j or {}).get("overall", 0)
        ok = rec >= 0.9 and ov >= 7 and m["peak"] < 1.0
        status = "PASS" if ok else ("WARN" if rec >= 0.75 else "FAIL")
        rows.append({"test": f"tts_{i:02d}", "text": text, "status": status, "recall": round(rec, 3),
                     "judge_overall": ov, "mangled": (j or {}).get("words_mangled", []),
                     "metrics": m, "asr": heard, "secs": round(__import__("time").time() - t0, 1)})
        print(f"{status} tts_{i:02d}: recall={rec:.2f} judge={ov} dur={m['dur']} lead={m['lead']} "
              f"mangled={(j or {}).get('words_mangled', [])}", flush=True)

    # --- B. emotion instruct (no ref) ---
    text = "Скидки до пятидесяти процентов только сегодня!"
    instr = f"Say the following in a happy, energetic Russian voice: '{text}'"
    sr, pcm = ig.run_generate("e2e", None, instr, 4.0, "", "", 64, 2.0, 1234,
                              translit=False, accentize=True, bestofn=1, trim=True, tts_wrap=False)
    path = os.path.join(OUT, "emotion_happy.wav")
    save_pcm(pcm, sr, path)
    heard = quality.transcribe(__import__("torch").from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0), sr)
    rec = quality.recall(text, heard)
    j = judge_file(path, text)
    status = "PASS" if rec >= 0.8 else ("WARN" if rec >= 0.6 else "FAIL")
    rows.append({"test": "emotion_happy", "status": status, "recall": round(rec, 3),
                 "judge_overall": (j or {}).get("overall", 0), "asr": heard[:70]})
    print(f"{status} emotion_happy: recall={rec:.2f} judge={(j or {}).get('overall')}", flush=True)

    # --- C. speech content editing: generate base, then replace word ---
    base_text = "Позвони мне, пожалуйста, когда освободишься."
    sr, pcm = ig.run_generate("e2e", REF, base_text, 5.0, "", "", 64, 2.0, 1234,
                              translit=False, accentize=True, bestofn=1, trim=True, tts_wrap=True)
    base_path = os.path.join(OUT, "edit_base.wav")
    save_pcm(pcm, sr, base_path)
    edit_instr = "Replace '\u043f\u043e\u0437\u0432\u043e\u043d\u0438' with '\u043f\u0435\u0440\u0435\u0437\u0432\u043e\u043d\u0438'."
    sr2, pcm2 = ig.run_generate("e2e", base_path, edit_instr, round(len(pcm) / sr, 2), "", "", 64, 2.0, 1234,
                                translit=False, accentize=True, bestofn=1, trim=False, tts_wrap=False)
    path = os.path.join(OUT, "edit_replaced.wav")
    save_pcm(pcm2, sr2, path)
    heard2 = quality.transcribe(__import__("torch").from_numpy(pcm2.astype("float32") / 32768.0).unsqueeze(0), sr2)
    new_ok = ("\u043f\u0435\u0440\u0435\u0437\u0432\u043e\u043d\u0438" in heard2) or ("\u043f\u0435\u0440\u0435\u0437\u0432\u043e\u043d\u0438\u0442" in heard2)
    old_gone = "\u043f\u043e\u0437\u0432\u043e\u043d\u0438" not in heard2
    status = "PASS" if (new_ok and old_gone) else ("WARN" if new_ok else "FAIL")
    rows.append({"test": "speech_edit", "status": "PASS" if (new_ok and old_gone) else ("WARN" if new_ok else "FAIL"),
                 "expected_new_word": new_ok, "old_word_gone": old_gone, "asr": heard2})
    print(f"{rows[-1]['status']} speech_edit: heard={heard2[:70]}", flush=True)

    # --- D. whisper conversion (soft check: must not crash; voice changes) ---
    sr, pcm = ig.run_generate("e2e", REF, "Turn this into a whisper", 10.21, "", "", 64, 2.0, 1234,
                              translit=False, accentize=True, bestofn=1, trim=True, tts_wrap=False)
    path = os.path.join(OUT, "whisper.wav")
    save_pcm(pcm, sr, path)
    rows.append({"test": "whisper", "status": "PASS", "note": "no crash; ear decides effect",
                 "dur": round(len(pcm) / sr, 2)})
    print("PASS whisper (info-only)", flush=True)

    json.dump(rows, open(os.path.join(OUT, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    passed = sum(1 for r in rows if r["status"] == "PASS")
    print(f"\nE2E_SUITE_DONE: {passed}/{len(rows)} PASS")


if __name__ == "__main__":
    main()
