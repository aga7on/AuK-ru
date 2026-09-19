# -*- coding: utf-8 -*-
"""AuK-ru v1.0 — канонический инференс-рецепт в одной точке входа.

Рецепт v1.0 (full_sha256.json _canonical):
  s7@5750 + Russian frontend (to_speakable) + best-of-N seeds (7,123,999)
  + composite reranker (sim+ASR+DNSMOS+loudness+pause − penalties)
  + normalize_rms + limit_peak + max_ref_seconds=30.

usage:
  python generate_v1.py --text "Встреча 21.09.2026 в 14:30." [--ref ref.wav] \
      [--emotion happy] [--best-of 3] [--out out.wav] [--device cuda:1]

Режимы:
  TTS       : --text, без --ref  → «Say the following in Russian...»
  clone     : --text --ref       → «Reproduce the reference voice and say in Russian: ...»
  clone+эмоция: --text --ref --emotion happy → «... with a happy tone: ...»
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

V1_CKPT = os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors")
V1_CONFIG = os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml")
SEEDS = (7, 123, 999)


def build_instruction(text, ref, emotion):
    if ref is None:
        return f"Say the following in Russian with clear, natural pronunciation: '{text}'"
    if emotion:
        return f"Reproduce the reference voice and say in Russian with a {emotion} tone: '{text}'"
    return f"Reproduce the reference voice and say in Russian: '{text}'"


def composite_score(x, sr, file_path, ref_emb, text):
    from ru_metrics import transcribe_path, text_metrics
    from rerank_composite import (W, loudness_score, pause_score, repetition_penalty,
                                  artifact_penalty, naturalness_score)
    heard, _ = transcribe_path(file_path)
    tm = text_metrics(text, heard)
    asr_ok = float(1 - min(tm["wer"], 1.0))
    nat = float(naturalness_score(x, sr))
    loud = float(loudness_score(x))
    pause = float(pause_score(x, sr))
    rep = float(repetition_penalty(heard))
    art = float(artifact_penalty(x))
    if ref_emb is not None:
        from speaker_embed import embed_path
        emb = embed_path(file_path)
        sim = float(np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-9))
        score = (W["sim"] * sim + W["asr"] * asr_ok + W["natural"] * nat
                 + W["loud"] * loud + W["pause"] * pause - rep - art)
    else:
        # TTS без референса: renormalize веса без sim (asr 0.40, nat 0.30, loud/pause 0.15)
        sim = None
        score = (0.40 * asr_ok + 0.30 * nat + 0.15 * loud + 0.15 * pause - rep - art)
    return float(score), {"sim": (round(sim, 4) if sim is not None else None),
                          "asr": round(asr_ok, 3), "nat": round(nat, 3),
                          "loud": round(loud, 3), "pause": round(pause, 3),
                          "rep": rep, "art": art, "heard": (heard or "")[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--emotion", default=None,
                    choices=[None, "happy", "sad", "angry", "fearful", "excited",
                             "whispering", "laughing", "surprised", "disgusted"])
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "v1_output"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--ckpt", default=V1_CKPT)
    ap.add_argument("--config", default=V1_CONFIG)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from auk.infer.ru_frontend import to_speakable

    os.makedirs(args.out, exist_ok=True)
    speak = to_speakable(args.text)
    instr = build_instruction(speak, args.ref, args.emotion)

    eng = AukInfer(config_path=args.config, ckpt_path=args.ckpt,
                   qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                   cpu_offload=True, device=args.device, dtype="bf16")

    gen_seconds = None
    content = [{"type": "text", "text": instr}]
    if args.ref:
        gen_seconds = float(sf.info(args.ref).duration) + 0.5
        content.append({"type": "audio", "audio": args.ref})
    else:
        gen_seconds = max(3.5, min(14.0, 1.0 + 0.09 * len(speak)))

    ref_emb = None
    if args.ref:
        from speaker_embed import embed_path
        ref_emb = embed_path(args.ref)

    n = max(1, min(args.best_of, len(SEEDS)))
    cands = []
    for i, seed in enumerate(SEEDS[:n]):
        audio, sr = eng.generate([{"role": "user", "content": content}],
                                 audio=args.ref, gen_seconds=gen_seconds,
                                 nfe=64, cfg_strength=2.0, seed=seed)
        audio = limit_peak(normalize_rms(audio))
        fp = os.path.join(args.out, f"cand_s{seed}.wav")
        save_audio(audio, sr, fp)
        if True:
            x, sr2 = sf.read(fp, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            score, det = composite_score(x, sr2, fp, ref_emb, speak.replace("+", ""))
        cands.append({"seed": seed, "file": fp, "score": score, **det})
        print(f"seed {seed}: score={score:.4f} sim={det.get('sim')} heard={str(det.get('heard'))[:60]}", flush=True)

    if len(cands) == 1:
        best = cands[0]
    else:
        best = max(cands, key=lambda c: c["score"])
    final = os.path.join(args.out, "v1_best.wav")
    import shutil
    shutil.copyfile(best["file"], final)
    print("BEST:", best["seed"], "->", final)
    if args.json:
        json.dump({"instruction": instr, "speakable": speak, "best": best, "cands": cands,
                   "ckpt": os.path.basename(args.ckpt)},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
