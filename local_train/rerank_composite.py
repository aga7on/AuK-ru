# -*- coding: utf-8 -*-
"""Composite reranker (v1.0 recipe / S11): выбор лучшего кандидата из best-of-N.

score = 0.35·sim + 0.25·asr_fidelity + 0.10·loudness + 0.10·pause_quality
        + 0.20·judge-free naturalness proxy (DNSMOS если доступен, иначе 0.5)
        − repetition_penalty − artifact_penalty

Компоненты (все локальные, без внешнего судьи):
  sim            — WeSpeaker cosine к референсу (speaker_embed);
  asr_fidelity   — 1 − WER против целевого текста (GigaAM через ru_metrics);
  loudness       — |RMS_dB − (−20)| / 20 (клиппинг → штраф в artifacts);
  pause_quality  — 1 при 2 паузах ≤1.2с, штраф за 0 или >4 пауз / паузу >2с;
  repetition     — повторы слов в ASR-транскрипте (n-gram ≥2 подряд) → 0.3;
  artifacts      — clip_ratio > 2e-4 или пустой/очень короткий выход → 0.3.

Веса — стартовые из ROADMAP S11; калибровка на human A/B (S13) позже.

usage:
  python rerank_composite.py --cands <dir> --ref <ref.wav> --text "<целевой текст>" [--json out.json]
  (кандидаты: все *.wav в dir; для нескольких текстов — по одному запуску)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

W = {"sim": 0.35, "asr": 0.25, "natural": 0.20, "loud": 0.10, "pause": 0.10}


def load_mono(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def loudness_score(x):
    act = x[np.abs(x) > 0.005]
    if len(act) == 0:
        return 0.0
    db = 20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12)
    return max(0.0, 1.0 - abs(db - (-20.0)) / 20.0)


def pause_score(x, sr):
    from pause_stats import rms_frames
    r = rms_frames(x)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    pauses, cur = 0, 0
    max_pause = 0.0
    for s in speech:
        if not s:
            cur += 1
        else:
            if cur:
                pauses += 1
                max_pause = max(max_pause, cur * 240 / sr)
            cur = 0
    if pauses == 0:
        return 0.7 if max_pause == 0 else 1.0
    if pauses > 4 or max_pause > 2.0:
        return 0.3
    return 1.0


def repetition_penalty(heard):
    words = re.findall(r"[а-яёa-z]+", (heard or "").lower())
    pen = 0.0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            pen += 0.3
    for i in range(2, len(words)):
        if words[i - 2:i] == words[i:i + 2] and len(words[i]) > 2:
            pen += 0.3
    return min(pen, 0.6)


def artifact_penalty(x):
    if len(x) < 0.3 * 16000:
        return 0.3
    clip = float(np.mean(np.abs(x) > 0.999))
    return 0.3 if clip > 2e-4 else 0.0


def naturalness_score(x, sr):
    """DNSMOS OVRL (speechmos) → 0..1; fallback 0.5 если модуль недоступен."""
    try:
        from speechmos import dnsmos as dnsmos_mod
        global _DNS
        if _DNS is None:
            _DNS = dnsmos_mod
        import numpy as _np
        wav16 = _np.interp(_np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)),
                           _np.arange(len(x)), x).astype("float32")
        d = _DNS.run(wav16, 16000)
        return max(0.0, min(float(d.get("ovrl_mos", 2.5)), 4.5)) / 4.5
    except Exception:
        return 0.5


_DNS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from ru_metrics import transcribe_path, text_metrics
    from speaker_embed import embed_path

    ref_e = embed_path(args.ref)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.cands, "*.wav"))):
        try:
            x, sr = load_mono(f)
            heard, asr_err = transcribe_path(f)
            tm = text_metrics(args.text, heard)
            sim = float(np.dot(ref_e, embed_path(f)) / (np.linalg.norm(ref_e) * np.linalg.norm(embed_path(f)) + 1e-9))
            row = {
                "file": os.path.basename(f),
                "sim": round(sim, 4),
                "asr": round(1.0 - min(tm["wer"], 1.0), 4),
                "natural": round(naturalness_score(x, sr), 3),
                "loud": round(loudness_score(x), 3),
                "pause": round(pause_score(x, sr), 3),
                "rep_pen": round(repetition_penalty(heard), 2),
                "art_pen": round(artifact_penalty(x), 2),
                "asr_error": asr_err,
                "heard": (heard or "")[:120],
            }
            row["score"] = round(float(W["sim"] * row["sim"] + W["asr"] * row["asr"]
                                 + W["natural"] * row["natural"] + W["loud"] * row["loud"]
                                 + W["pause"] * row["pause"] - row["rep_pen"] - row["art_pen"]), 4)
            for k in ("sim", "asr", "loud", "pause", "rep_pen", "art_pen"):
                row[k] = float(row[k])
            rows.append(row)
        except Exception as e:
            rows.append({"file": os.path.basename(f), "score": None, "error": type(e).__name__})

    rows_ok = [r for r in rows if r.get("score") is not None]
    best = max(rows_ok, key=lambda r: r["score"]) if rows_ok else None
    print(json.dumps({"best": best, "n": len(rows)}, ensure_ascii=False, indent=1)[:600])
    if args.json:
        json.dump({"best": best["file"] if best else None, "rows": rows},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("saved", args.json)


if __name__ == "__main__":
    main()
