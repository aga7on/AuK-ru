# -*- coding: utf-8 -*-
"""S14b: nfe=16 draft-mode — RTF и качество против nfe=64 (эталон) и nfe=32.

10 текстов (5 clone из eval_pack + 5 tts) × nfe {64, 32, 16} × seed 7.
Метрики: RTF, WER (GigaAM), sim (WeSpeaker, clone).
Гейт draft-режима: ΔWER ≤ +0.03 и Δsim ≥ −0.03 против nfe64 → режим публикуется
как «черновой» (быстрый предпросмотр), иначе отклоняется.
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "nfe16_check")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    os.makedirs(OUT, exist_ok=True)
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from ru_metrics import transcribe_path, text_metrics
    from speaker_embed import embed_path

    eng = AukInfer(config_path=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"),
                   ckpt_path=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device="cuda:0", dtype="bf16")

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    items = [e for e in pack if e.get("kind") == "clone"][:5] + [e for e in pack if e.get("kind") == "tts"][:5]

    rows = []
    for it in items:
        ref = it.get("ref")
        ref_e = embed_path(ref) if ref else None
        for nfe in (64, 32, 16):
            content = [{"type": "text", "text": it["instruction"]}]
            kw = {}
            if ref:
                content.append({"type": "audio", "audio": ref})
                kw["audio"] = ref
                gs = float(sf.info(ref).duration) + 0.5
            else:
                gs = float(it.get("gen_seconds") or 6.0)
            fp = os.path.join(OUT, f"{it['id']}_nfe{nfe}.wav")
            t0 = time.time()
            audio, sr = eng.generate([{"role": "user", "content": content}],
                                     gen_seconds=gs, nfe=nfe, cfg_strength=2.0, seed=7, **kw)
            gen_s = time.time() - t0
            audio = limit_peak(normalize_rms(audio))
            save_audio(audio, sr, fp)
            heard, _ = transcribe_path(fp)
            tm = text_metrics(it.get("text") or "", heard)
            x, sr2 = sf.read(fp, dtype="float32")
            dur = x.shape[-1] / sr2
            sim = round(cos(ref_e, embed_path(fp)), 4) if ref_e is not None else None
            rows.append({"id": it["id"], "kind": it["kind"], "nfe": nfe, "gen_s": round(gen_s, 2),
                         "dur": round(dur, 2), "rtf": round(gen_s / max(dur, 1e-6), 3),
                         "wer": round(float(tm["wer"]), 3), "sim": sim})
            print(rows[-1], flush=True)

    json.dump(rows, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    import statistics as st
    print("--- summary ---")
    for nfe in (64, 32, 16):
        rs = [r for r in rows if r["nfe"] == nfe]
        rtfs = [r["rtf"] for r in rs]
        wers = [r["wer"] for r in rs if r["kind"] == "clone"]
        sims = [r["sim"] for r in rs if r["sim"] is not None]
        print(f"nfe{nfe}: RTF med {st.median(rtfs):.3f} | clone WER med {st.median(wers):.3f} | clone sim med {st.median(sims):.4f}")
    base = {r["id"]: r for r in rows if r["nfe"] == 64}
    d16 = [r for r in rows if r["nfe"] == 16 and r["kind"] == "clone"]
    dw = st.median([r["wer"] - base[r["id"]]["wer"] for r in d16])
    ds = st.median([r["sim"] - base[r["id"]]["sim"] for r in d16])
    ok = dw <= 0.03 and ds >= -0.03
    print(f"NFE16 verdict: dWER {dw:+.3f} dsim {ds:+.4f} -> {'DRAFT-MODE OK' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
