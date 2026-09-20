# -*- coding: utf-8 -*-
"""S14b: nfe=64 vs nfe=32 — деградирует ли качество при ускорении?

10 текстов (5 tts_hard + 5 clone из eval_pack) × 2 nfe × seed 7 → GigaAM WER + WeSpeaker sim.
Гейт: nfe32 не хуже nfe64 (ΔWER ≤ +0.02, Δsim ≥ −0.02 median).

usage: python nfe_ab_check.py [--device cuda:0]
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "nfe_ab")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from ru_metrics import transcribe_path, text_metrics
    from speaker_embed import embed_path

    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"),
                   ckpt_path=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    clones = [e for e in pack if e.get("kind") == "clone"][:5]
    tts = [e for e in pack if e.get("kind") == "tts"][:5]
    items = clones + tts

    rows = []
    for it in items:
        for nfe in (64, 32):
            content = [{"type": "text", "text": it["instruction"]}]
            kw = {}
            ref = it.get("ref")
            if ref:
                content.append({"type": "audio", "audio": ref})
                kw["audio"] = ref
            fp = os.path.join(OUT, f"{it['id']}_nfe{nfe}.wav")
            audio, sr = eng.generate([{"role": "user", "content": content}],
                                     gen_seconds=float(it.get("gen_seconds") or sf.info(ref).duration + 0.5),
                                     nfe=nfe, cfg_strength=2.0, seed=7, **kw)
            audio = limit_peak(normalize_rms(audio))
            save_audio(audio, sr, fp)
            heard, _ = transcribe_path(fp)
            tm = text_metrics(it.get("text") or "", heard)
            sim = None
            if ref:
                sim = round(cos(embed_path(ref), embed_path(fp)), 4)
            rows.append({"id": it["id"], "kind": it["kind"], "nfe": nfe,
                         "wer": round(float(tm["wer"]), 3), "sim": sim})
            print(rows[-1], flush=True)

    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    import statistics as st
    w64 = [r["wer"] for r in rows if r["nfe"] == 64]
    w32 = [r["wer"] for r in rows if r["nfe"] == 32]
    s64 = [r["sim"] for r in rows if r["nfe"] == 64 and r["sim"] is not None]
    s32 = [r["sim"] for r in rows if r["nfe"] == 32 and r["sim"] is not None]
    dw = st.median(w32) - st.median(w64)
    ds = (st.median(s32) - st.median(s64)) if s64 and s32 else None
    print(f"NFE_AB: wer median 64={st.median(w64):.3f} 32={st.median(w32):.3f} (delta {dw:+.3f})")
    if ds is not None:
        print(f"NFE_AB: sim median 64={st.median(s64):.4f} 32={st.median(s32):.4f} (delta {ds:+.4f})")
    ok = (dw <= 0.02) and (ds is None or ds >= -0.02)
    print("VERDICT:", "nfe32 acceptable" if ok else "nfe32 degrades — keep nfe64")


if __name__ == "__main__":
    main()
