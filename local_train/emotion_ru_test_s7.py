"""Тест эмоций для произвольного merged-чекпойнта (копия emotion_ru_test.py, параметризованная).
5 эмоций (happy, sad, angry, fearful, excited) × 3 русские фразы × 2 сида × 2 голоса (male/female).
Использование:
  python emotion_ru_test_s7.py --ckpt <merged.safetensors> --config <config.yaml> --out <dir>
"""
import argparse
import json
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio

AUK = r"G:\AI\AuK"

EMOTIONS = ["happy", "sad", "angry", "fearful", "excited"]
PHRASES = [
    "привет, как у тебя дела сегодня?",
    "я не могу найти свои ключи уже целый час",
    "завтра мы поедем на дачу и приготовим шашлык",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    REFS = {"male": None, "female": None}
    for e in pack:
        if e.get("kind") == "clone" and e["id"] == "clone08":
            REFS["male"] = e["ref"]
        if e.get("kind") == "clone" and e["id"] == "clone01":
            REFS["female"] = e["ref"]
    print("refs:", {k: os.path.basename(v) if v else None for k, v in REFS.items()}, flush=True)

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")
    results = []
    t0 = time.time()
    for voice, ref in REFS.items():
        for emo in EMOTIONS:
            for pi, text in enumerate(PHRASES):
                for seed in (7, 999):
                    instr = f"Reproduce the reference voice and say in Russian with a {emo} tone: '{text}'"
                    out = os.path.join(OUT, f"{voice}_{emo}_p{pi}_s{seed}.wav")
                    content = [{"type": "text", "text": instr},
                               {"type": "audio", "audio": ref}]
                    try:
                        audio, sr = eng.generate([{"role": "user", "content": content}], audio=ref,
                                                 gen_seconds=float(sf.info(ref).duration) + 0.5,
                                                 nfe=64, cfg_strength=2.0, seed=seed)
                        from auk.infer.quality import normalize_rms, limit_peak
                        audio = limit_peak(normalize_rms(audio))
                        save_audio(audio, sr, out)
                        results.append({"id": f"{voice}_{emo}_p{pi}_s{seed}", "voice": voice,
                                        "emotion": emo, "text": text, "seed": seed,
                                        "ref": ref, "file": out, "status": "ok"})
                    except Exception as ex:
                        results.append({"id": f"{voice}_{emo}_p{pi}_s{seed}", "voice": voice,
                                        "emotion": emo, "text": text, "seed": seed,
                                        "ref": ref, "file": None, "status": f"error {type(ex).__name__}"})
                    print(f"[{len(results)}] {results[-1]['id']} {results[-1]['status']} ({(time.time()-t0)/60:.1f}m)", flush=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"EMOTION_RU_DONE ok={ok}/{len(results)} out={OUT}")


if __name__ == "__main__":
    main()