"""Build a filtered, gender-balanced Russian JSONL dataset for AuK LoRA fine-tune.

Stages:
  A. metadata filter of ru_train_aligned.jsonl -> candidates.jsonl
  B. per-clip audio metrics (DNSMOS quality, F0/gender, silence, clipping) -> metrics.jsonl (resumable)
  C. selection: quality + gender balance -> selected.jsonl
  D. text pipeline: RuAccent stress -> ru_to_latin -> instruction templates -> train.jsonl / val.jsonl
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

AUK_SRC = r"G:\AI\AuK\src"
if AUK_SRC not in sys.path:
    sys.path.insert(0, AUK_SRC)

CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LETTER_RE = re.compile(r"[a-zа-яё]", re.IGNORECASE)


def _f0_median(x, sr):
    """Median F0 over voiced frames via FFT autocorrelation (numpy only)."""
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    if len(x) < frame * 2:
        return 0.0
    n_frames = 1 + (len(x) - frame) // hop
    win = np.hanning(frame).astype(np.float32)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    f0s = []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame] * win
        if np.sqrt(np.mean(seg * seg)) < 1e-3:
            continue
        seg = seg - seg.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * frame)) ** 2)[:max_lag + 1]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        band = ac[min_lag:max_lag + 1]
        if len(band) == 0:
            continue
        peak = float(band.max())
        if peak < 0.35:
            continue
        lag = min_lag + int(np.argmax(band))
        f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def _analyze_one(path):
    from speechmos import dnsmos as dnsmos_mod

    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 24000:
            idx = np.linspace(0, len(x) - 1, int(len(x) * 24000 / sr))
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
            sr = 24000
        dur = len(x) / sr
        peak = float(np.abs(x).max()) if len(x) else 0.0
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        clipped = float(np.mean(np.abs(x) > 0.985))
        frame = int(0.02 * sr)
        nf = max(1, len(x) // frame)
        fr = x[: nf * frame].reshape(nf, frame)
        energy = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
        silence = float(np.mean(energy < 0.01))
        f0 = _f0_median(x, sr)
        wav16 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)),
                          np.arange(len(x)), x).astype(np.float32)
        d = dnsmos_mod.run(wav16, 16000)
        ovrl = float(d.get("ovrl_mos", d.get("p835_ovrl", 0.0)))
        sig = float(d.get("sig_mos", d.get("p835_sig", 0.0)))
        bak = float(d.get("bak_mos", d.get("p835_bak", 0.0)))
        return {
            "path": path, "dur": round(dur, 3), "f0": round(f0, 1), "peak": round(peak, 4),
            "rms": round(rms, 5), "clipped": round(clipped, 5), "silence": round(silence, 3),
            "ovrl": round(ovrl, 3), "sig": round(sig, 3), "bak": round(bak, 3), "ok": True,
        }
    except Exception as exc:
        return {"path": path, "ok": False, "err": f"{type(exc).__name__}: {exc}"}


def _dnsmos_import():
    from speechmos import dnsmos
    return dnsmos


def stage_a(manifest, out_dir, n_candidates, seed=7):
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    if os.path.exists(cand_path):
        rows = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
        if len(rows) >= n_candidates:
            print(f"[A] using existing candidates: {len(rows)}")
            return rows[:n_candidates]
        print(f"[A] existing candidates {len(rows)} < requested {n_candidates} -> rebuilding")
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            dur = r.get("duration")
            text = str(r.get("transcript", "")).strip()
            path = r.get("path")
            if not (path and isinstance(dur, (int, float)) and 1.2 <= dur <= 12.0):
                continue
            if not (10 <= len(text) <= 200) or not path.lower().endswith(".wav"):
                continue
            if re.search(r"\d", text) or "http" in text.lower() or len(text.split()) < 2:
                continue
            letters = LETTER_RE.findall(text)
            if not letters or len(CYR_RE.findall(text)) / len(letters) < 0.8:
                continue
            if not os.path.exists(path):
                continue
            rows.append({"path": path, "dur": float(dur), "text": text})
    random.Random(seed).shuffle(rows)
    rows = rows[:n_candidates]
    with open(cand_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[A] candidates: {len(rows)} ({sum(r['dur'] for r in rows)/3600:.1f} h)")
    return rows


def stage_b(out_dir, rows, workers):
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    done = set()
    if os.path.exists(metrics_path):
        for line in open(metrics_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                pass
    todo = [r["path"] for r in rows if r["path"] not in done]
    print(f"[B] metrics: done={len(done)}, todo={len(todo)}")
    if not todo:
        return
    t0 = time.time()
    with open(metrics_path, "a", encoding="utf-8") as fout, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_analyze_one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                fout.flush()
                rate = i / (time.time() - t0)
                print(f"[B] {i}/{len(todo)} ({rate:.1f} clip/s, eta {(len(todo)-i)/max(rate,1e-6)/60:.1f} min)")
    print(f"[B] done in {(time.time()-t0)/60:.1f} min")


def stage_c(out_dir, target_clips, max_dur_hours, min_ovrl=3.0, min_sig=2.8):
    metrics = []
    with open(os.path.join(out_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok") and r.get("dur", 0) >= 1.2 and r.get("ovrl", 0) >= min_ovrl and r.get("sig", 0) >= min_sig \
                    and r.get("silence", 1) <= 0.45 and r.get("peak", 0) <= 0.995 and r.get("rms", 0) >= 0.01 \
                    and r.get("clipped", 1) <= 0.02:
                if 60 <= r["f0"] < 158:
                    r["gender"] = "m"
                elif r["f0"] > 178:
                    r["gender"] = "f"
                else:
                    continue
                metrics.append(r)
    cand_by_path = {}
    with open(os.path.join(out_dir, "candidates.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cand_by_path[r["path"]] = r
    for r in metrics:
        c = cand_by_path.get(r["path"], {})
        r["text"] = c.get("text", "")
    metrics = [r for r in metrics if r["text"]]
    males = sorted([r for r in metrics if r["gender"] == "m"], key=lambda r: -r["ovrl"])
    females = sorted([r for r in metrics if r["gender"] == "f"], key=lambda r: -r["ovrl"])
    per = target_clips // 2
    sel = males[:per] + females[:per]
    sel.sort(key=lambda r: r["path"])
    max_clips = int(max_dur_hours * 3600 / max(np.mean([r["dur"] for r in sel]) if sel else 1, 1))
    if len(sel) > max_clips:
        sel = sel[:max_clips]
    selected_path = os.path.join(out_dir, "selected.jsonl")
    with open(selected_path, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[C] selected {len(sel)} clips ({sum(r['dur'] for r in sel)/3600:.1f} h) | "
          f"male={sum(1 for r in sel if r['gender']=='m')} female={sum(1 for r in sel if r['gender']=='f')} | "
          f"ovrl m={np.mean([r['ovrl'] for r in sel if r['gender']=='m']):.2f} "
          f"f={np.mean([r['ovrl'] for r in sel if r['gender']=='f']):.2f}")
    return sel


TEMPLATES = [
    "Say the following in Russian with clear, natural pronunciation: '{t}'",
    "Speak the following Russian text aloud in a natural voice: '{t}'",
    "Pronounce the following in Russian: '{t}'",
    "Read the following Russian sentence out loud: '{t}'",
]


def stage_d(out_dir, sel, val_clips=256, mix="cyr_stress:0.65,cyr_plain:0.25,translit:0.10"):
    from auk.infer.ru_translit import ru_to_latin
    from ruaccent import RUAccent

    reps, fracs = [], []
    for part in mix.split(","):
        name, _, val = part.partition(":")
        reps.append(name.strip())
        fracs.append(float(val))
    total = sum(fracs) or 1.0
    fracs = [f / total for f in fracs]
    cum, acc = [], 0.0
    for f in fracs:
        acc += f
        cum.append(acc)

    def rep_for(i):
        x = ((i % 100) + 0.5) / 100.0
        for name, c in zip(reps, cum):
            if x <= c:
                return name
        return reps[-1]

    accentizer = RUAccent()
    accentizer.load(omograph_model_size="turbo2", use_dictionary=True, device="CPU")

    rng = random.Random(42)
    rng.shuffle(sel)
    val = sel[:val_clips]
    train = sel[val_clips:]

    def prep(rows, out_name):
        counts = {name: 0 for name in reps}
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                text = " ".join(r["text"].split())
                rep = rep_for(i)
                counts[rep] += 1
                if rep == "cyr_stress":
                    body = accentizer.process_all(text)
                elif rep == "cyr_plain":
                    body = text
                else:
                    body = ru_to_latin(accentizer.process_all(text))
                instr = TEMPLATES[i % len(TEMPLATES)].format(t=body)
                row = {
                    "duration": round(r["dur"], 3),
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": instr}]},
                        {"role": "assistant", "content": [{"type": "audio", "audio_url": r["path"]}]},
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return counts

    c1 = prep(train, "train.jsonl")
    c2 = prep(val, "val.jsonl")
    print(f"[D] train={len(train)} ({sum(r['dur'] for r in train)/3600:.1f} h, {c1}) | "
          f"val={len(val)} ({sum(r['dur'] for r in val)/3600:.1f} h, {c2})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"G:\AI\kyutai-ru\data\ru_train_aligned.jsonl")
    ap.add_argument("--out_dir", default=r"G:\AI\AuK\local_train\data")
    ap.add_argument("--candidates", type=int, default=40000)
    ap.add_argument("--target_clips", type=int, default=24000)
    ap.add_argument("--max_hours", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_ovrl", type=float, default=3.0)
    ap.add_argument("--min_sig", type=float, default=2.8)
    ap.add_argument("--mix", default="cyr_stress:0.65,cyr_plain:0.25,translit:0.10")
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "c", "d"])
    ap.add_argument("--limit", type=int, default=0, help="debug: limit candidates")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = stage_a(args.manifest, args.out_dir, args.candidates)
    if args.limit:
        rows = rows[: args.limit]
    if args.stage in ("all", "b"):
        stage_b(args.out_dir, rows, args.workers)
    if args.stage in ("all", "c", "d"):
        sel = stage_c(args.out_dir, args.target_clips, args.max_hours, args.min_ovrl, args.min_sig)
        if args.stage in ("all", "d"):
            stage_d(args.out_dir, sel, mix=args.mix)


if __name__ == "__main__":
    main()
