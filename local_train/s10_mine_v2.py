# -*- coding: utf-8 -*-
"""S10 mining v2: preference-пары на HELD-OUT текстах (hard_cases_ru.jsonl)
и референсах вне train-пула (как clone100). Чистая основа для RFT/DPO:
ни тексты, ни рефы не встречаются в v5/v7/v8/v9 миксах.

50 текстов (hard_cases) × 4 seeds на v1.0 (s7@5750) → composite-скоринг →
пары chosen/rejected (gap ≥ 0.05) → preference/pairs_mined_v2.jsonl.

usage: python s10_mine_v2.py [--n_texts 50] [--device cuda:1]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "s10_mine_v2")
SEEDS = (7, 123, 999, 2026)
CORPUS = r"G:\AI\kyutai-ru\data\ru_wav_mfa"
TRAIN_POOLS = [
    os.path.join(AUK, "local_train", "data_s2_full", "v2_after_identity", "train.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_full", "v5_s5_mix", "train.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_full", "v7_s7_mix", "train.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_full", "v9_s8b_mix", "train.jsonl"),
]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pick_refs(n, seed=10):
    pool = set()
    for tp in TRAIN_POOLS:
        if not os.path.exists(tp):
            continue
        for l in open(tp, encoding="utf-8"):
            r = json.loads(l)
            for m in r["messages"]:
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "audio":
                        a = c.get("audio") or c.get("audio_url")
                        if a:
                            pool.add(os.path.normcase(a))
    rng = random.Random(seed)
    corpus = [os.path.join(CORPUS, f) for f in os.listdir(CORPUS) if f.endswith(".wav")]
    rng.shuffle(corpus)
    out = []
    for f in corpus:
        if os.path.normcase(f) in pool:
            continue
        try:
            d = sf.info(f).duration
        except Exception:
            continue
        if 2.5 <= d <= 8.0:
            out.append(f)
        if len(out) >= n:
            break
    print(f"refs picked: {len(out)} (pool excluded {len(pool)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_texts", type=int, default=50)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ckpt", default=os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"))
    ap.add_argument("--config", default=os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    hc = [json.loads(l) for l in open(os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru.jsonl"), encoding="utf-8") if l.strip()]
    # приоритет: mined_s7_fail (реальные провалы) + long_word + phonetic-тяжёлые
    prio = {"mined_s7_fail": 0, "long_word_cluster": 1, "phone": 2, "abbr_name": 3, "code_switch": 4}
    hc.sort(key=lambda r: prio.get(r["category"], 5))
    texts = hc[: args.n_texts]
    refs = pick_refs(len(texts))

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from auk.infer.ru_frontend import to_speakable
    from ru_metrics import transcribe_path, text_metrics
    from speaker_embed import embed_path
    from rerank_composite import (W, loudness_score, pause_score, repetition_penalty,
                                  artifact_penalty, naturalness_score)

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    results = []
    t0 = time.time()
    for ti, (h, ref) in enumerate(zip(texts, refs)):
        speak = to_speakable(h["text"])
        expected = speak.replace("+", "")
        instr = f"Reproduce the reference voice and say in Russian: '{speak}'"
        ref_e = embed_path(ref)
        cands = []
        for seed in SEEDS:
            fp = os.path.join(OUT, f"t{ti:03d}_s{seed}.wav")
            try:
                audio, sr = eng.generate([{"role": "user", "content": [
                    {"type": "text", "text": instr}, {"type": "audio", "audio": ref}]}],
                    audio=ref, gen_seconds=float(sf.info(ref).duration) + 0.5,
                    nfe=64, cfg_strength=2.0, seed=seed)
                audio = limit_peak(normalize_rms(audio))
                save_audio(audio, sr, fp)
                x, sr2 = sf.read(fp, dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                heard, _ = transcribe_path(fp)
                tm = text_metrics(expected, heard)
                emb = embed_path(fp)
                row = {"text_id": f"t{ti:03d}", "seed": seed, "file": fp, "ref": ref,
                       "text": h["text"], "category": h["category"],
                       "sim": round(float(cos(ref_e, emb)), 4),
                       "asr": round(float(1 - min(tm["wer"], 1.0)), 3),
                       "nat": round(float(naturalness_score(x, sr2)), 3),
                       "loud": round(float(loudness_score(x)), 3),
                       "pause": round(float(pause_score(x, sr2)), 3),
                       "rep": round(float(repetition_penalty(heard)), 2),
                       "art": round(float(artifact_penalty(x)), 2)}
                row["score"] = round(float(W["sim"] * row["sim"] + W["asr"] * row["asr"]
                                           + W["natural"] * row["nat"] + W["loud"] * row["loud"]
                                           + W["pause"] * row["pause"] - row["rep"] - row["art"]), 4)
                json.dumps(row)  # dry-run (ERRORS.MD #4)
                cands.append(row)
            except Exception as e:
                cands.append({"text_id": f"t{ti:03d}", "seed": seed, "error": type(e).__name__})
        results.append({"text_id": f"t{ti:03d}", "ref": ref, "text": h["text"],
                        "category": h["category"], "cands": cands})
        json.dump(results, open(os.path.join(OUT, "results_partial.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{ti+1}/{len(texts)}] t{ti:03d} scores={[c.get('score') for c in cands]} ({(time.time()-t0)/60:.1f}m)", flush=True)

    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    pairs = []
    for r in results:
        ok = [c for c in r["cands"] if c.get("score") is not None]
        if len(ok) < 2:
            continue
        b = max(ok, key=lambda c: c["score"])
        w = min(ok, key=lambda c: c["score"])
        if b["score"] - w["score"] < 0.05:
            continue
        pairs.append({"pair_id": f"mined2:{r['text_id']}", "source": "s10_mine_v2_holdout",
                      "prompt_id": r["text_id"], "ref": r["ref"], "text": r["text"],
                      "category": r["category"],
                      "chosen": b["file"], "rejected": w["file"],
                      "chosen_score": b["score"], "rejected_score": w["score"],
                      "metric": "composite_v1", "chosen_seed": b["seed"], "rejected_seed": w["seed"],
                      "chosen_sim": b["sim"], "rejected_sim": w["sim"]})
    pref = os.path.join(AUK, "local_train", "preference")
    os.makedirs(pref, exist_ok=True)
    with open(os.path.join(pref, "pairs_mined_v2.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("MINE_V2_DONE texts=%d pairs=%d" % (len(results), len(pairs)))


if __name__ == "__main__":
    main()
