# -*- coding: utf-8 -*-
"""S10 rescore: считает composite по уже сгенерированным wav (local_tests/s10_mine),
без повторной генерации. Фикс: все метрики приводятся к float().

usage: python s10_rescore.py
Выход: local_tests/s10_mine/results.json + local_train/preference/pairs_mined_v1.jsonl
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

from ru_metrics import transcribe_path, text_metrics
from speaker_embed import embed_path
from rerank_composite import (W, loudness_score, pause_score, repetition_penalty,
                              artifact_penalty, naturalness_score)

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "s10_mine")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def composite(sim, asr_ok, nat, loud, pause, rep, art):
    return float(W["sim"] * sim + W["asr"] * asr_ok + W["natural"] * nat
                 + W["loud"] * loud + W["pause"] * pause - rep - art)


def main():
    src = json.load(open(os.path.join(AUK, "local_tests", "s7_5750_clone100", "results.json"), encoding="utf-8"))
    prompts = [r for r in src if r.get("status") == "ok"][:25]

    results = []
    for pi, p in enumerate(prompts):
        ref, text = p["ref"], p["text"]
        ref_e = embed_path(ref)
        cands = []
        for seed in (7, 123, 999, 2026):
            fp = os.path.join(OUT, f"p{pi:03d}_s{seed}.wav")
            if not os.path.exists(fp):
                continue
            x, sr2 = sf.read(fp, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            heard, _ = transcribe_path(fp)
            tm = text_metrics(text, heard)
            row = {"prompt_id": f"p{pi:03d}", "seed": seed, "file": fp, "ref": ref,
                   "text": text,
                   "sim": round(float(cos(ref_e, embed_path(fp))), 4),
                   "asr": round(float(1 - min(tm["wer"], 1.0)), 3),
                   "nat": round(float(naturalness_score(x, sr2)), 3),
                   "loud": round(float(loudness_score(x)), 3),
                   "pause": round(float(pause_score(x, sr2)), 3),
                   "rep": round(float(repetition_penalty(heard)), 2),
                   "art": round(float(artifact_penalty(x)), 2)}
            row["score"] = round(composite(row["sim"], row["asr"], row["nat"],
                                           row["loud"], row["pause"], row["rep"], row["art"]), 4)
            cands.append(row)
        results.append({"prompt_id": f"p{pi:03d}", "ref": ref, "text": text, "cands": cands})
        print(f"[{pi+1}/{len(prompts)}] scores={[c.get('score') for c in cands]}", flush=True)

    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

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
    pref = os.path.join(AUK, "local_train", "preference")
    os.makedirs(pref, exist_ok=True)
    with open(os.path.join(pref, "pairs_mined_v1.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    import statistics as st
    first_scores, best_scores = [], []
    for r in results:
        ok = [c for c in r["cands"] if c.get("score") is not None]
        if not ok:
            continue
        f7 = next((c for c in ok if c["seed"] == 7), None)
        if f7:
            first_scores.append(f7["score"])
        best_scores.append(max(c["score"] for c in ok))
    print("RESCORE_DONE prompts=%d pairs=%d" % (len(results), len(pairs)))
    if first_scores:
        print("composite first-seed median %.4f vs best-of-4 %.4f (gap %+.4f)"
              % (st.median(first_scores), st.median(best_scores),
                 st.median(best_scores) - st.median(first_scores)))


if __name__ == "__main__":
    main()
