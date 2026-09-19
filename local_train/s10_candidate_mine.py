# -*- coding: utf-8 -*-
"""S10: N>1 candidate mining на clone100-промптах (v1.0 = s7@5750, cuda:1).

25 промптов × 4 seeds (7,123,999,2026) → кандидаты; скоринг composite
(WeSpeaker sim + GigaAM ASR-fidelity + DNSMOS + loudness + pause − penalties);
пары: chosen=argmax, rejected=argmin при composite-gap ≥ 0.05.

usage: python s10_candidate_mine.py [--n_prompts 25] [--device cuda:1]
Выход: local_tests\s10_mine\{wav,results.json} + local_train\preference\pairs_mined_v1.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from auk.infer.infer_auk import AukInfer, save_audio
from auk.infer.quality import normalize_rms, limit_peak
from ru_metrics import transcribe_path, text_metrics
from speaker_embed import embed_path
from rerank_composite import (W, loudness_score, pause_score, repetition_penalty,
                              artifact_penalty, naturalness_score)

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "s10_mine")
SEEDS = (7, 123, 999, 2026)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def composite(sim, asr_ok, nat, loud, pause, rep, art):
    return (W["sim"] * sim + W["asr"] * asr_ok + W["natural"] * nat
            + W["loud"] * loud + W["pause"] * pause - rep - art)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=25)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"))
    ap.add_argument("--config", default=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    src = json.load(open(os.path.join(AUK, "local_tests", "s7_5750_clone100", "results.json"), encoding="utf-8"))
    prompts = [r for r in src if r.get("status") == "ok"][: args.n_prompts]
    print("prompts:", len(prompts), flush=True)

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    results = []
    t0 = time.time()
    for pi, p in enumerate(prompts):
        ref, text = p["ref"], p["text"]
        ref_e = embed_path(ref)
        instr = p.get("instruction") or f"Reproduce the reference voice and say in Russian: '{text}'"
        cands = []
        for seed in SEEDS:
            fp = os.path.join(OUT, f"p{pi:03d}_s{seed}.wav")
            try:
                audio, sr = eng.generate([{"role": "user", "content": [
                    {"type": "text", "text": instr}, {"type": "audio", "audio": ref}]}],
                    audio=ref, gen_seconds=float(sf.info(ref).duration) + 0.5,
                    nfe=64, cfg_strength=2.0, seed=seed)
                audio = limit_peak(normalize_rms(audio))
                save_audio(audio, sr, fp)
                x, sr2 = sf.read(fp, dtype="float32")
                heard, _ = transcribe_path(fp)
                tm = text_metrics(text, heard)
                sim = cos(ref_e, embed_path(fp))
                row = {"prompt_id": f"p{pi:03d}", "seed": seed, "file": fp, "ref": ref,
                       "text": text, "sim": round(float(sim), 4),
                       "asr": round(float(1 - min(tm["wer"], 1.0)), 3),
                       "nat": round(float(naturalness_score(x, sr2)), 3),
                       "loud": round(float(loudness_score(x)), 3),
                       "pause": round(float(pause_score(x, sr2)), 3),
                       "rep": round(float(repetition_penalty(heard)), 2),
                       "art": round(float(artifact_penalty(x)), 2)}
                row["score"] = round(float(composite(row["sim"], row["asr"], row["nat"],
                                                     row["loud"], row["pause"], row["rep"], row["art"])), 4)
                json.dumps(row)  # dry-run сериализации (ERRORS.MD #4)
                cands.append(row)
            except Exception as e:
                cands.append({"prompt_id": f"p{pi:03d}", "seed": seed, "error": type(e).__name__})
        results.append({"prompt_id": f"p{pi:03d}", "ref": ref, "text": text, "cands": cands})
        print(f"[{pi+1}/{len(prompts)}] {results[-1]['prompt_id']} scores={[c.get('score') for c in cands]} ({(time.time()-t0)/60:.1f}m)", flush=True)

    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # pairs: gap >= 0.05
    pairs = []
    for r in results:
        ok = [c for c in r["cands"] if c.get("score") is not None]
        if len(ok) < 2:
            continue
        b = max(ok, key=lambda c: c["score"])
        w = min(ok, key=lambda c: c["score"])
        if b["score"] - w["score"] < 0.05:
            continue
        pairs.append({"pair_id": f"mined:{r['prompt_id']}", "source": "s10_candidate_mine_v1",
                      "prompt_id": r["prompt_id"], "ref": r["ref"], "text": r["text"],
                      "chosen": b["file"], "rejected": w["file"],
                      "chosen_score": b["score"], "rejected_score": w["score"],
                      "metric": "composite_v1", "chosen_seed": b["seed"], "rejected_seed": w["seed"],
                      "chosen_sim": b["sim"], "rejected_sim": w["sim"]})
    pref_dir = os.path.join(AUK, "local_train", "preference")
    os.makedirs(pref_dir, exist_ok=True)
    with open(os.path.join(pref_dir, "pairs_mined_v1.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # first-shot stats
    first_scores = []
    best_scores = []
    for r in results:
        ok = [c for c in r["cands"] if c.get("score") is not None]
        if not ok:
            continue
        f7 = next((c for c in ok if c["seed"] == 7), None)
        if f7:
            first_scores.append(f7["score"])
        best_scores.append(max(c["score"] for c in ok))
    import statistics as st
    print("MINE_DONE prompts=%d pairs=%d" % (len(results), len(pairs)))
    if first_scores:
        print("composite first-seed median %.4f vs best-of-4 %.4f (gap %+.4f)"
              % (st.median(first_scores), st.median(best_scores),
                 st.median(best_scores) - st.median(first_scores)))


if __name__ == "__main__":
    main()
