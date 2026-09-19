"""Autonomous controller for the RU LoRA run.

Phases:
  0. baseline deep-eval of the un-tuned model (GPU1)
  1. loop: deep-eval of new checkpoints (merge -> generate -> Qwen judge), track best,
     stop training when quality is enough / degrading / diverging, then finalize best.
"""
import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

AUK = r"G:\AI\AuK"
RUN_DIR = os.path.join(AUK, "local_train", "run_ru")
EVAL_ROOT = os.path.join(RUN_DIR, "evals")
MERGED_DIR = os.path.join(RUN_DIR, "merged")
CONTROL_LOG = os.path.join(RUN_DIR, "control.jsonl")
TRAIN_LOG = r"G:\AI\_tmp\train_ru.log"
BASE_CKPT = os.path.join(AUK, "ckpts", "AuK", "auk_base.safetensors")
PY = os.path.join(AUK, ".venv", "Scripts", "python.exe")
LOCAL = os.path.join(AUK, "local_train")

MIN_UPDATES_BETWEEN_EVALS = 500
MIN_MINUTES_BETWEEN_EVALS = 25
MIN_STOP_UPDATE = 3000
MAX_UPDATES = 7000
MAX_HOURS = 15.0
ENOUGH_OVERALL = 8.3
ENOUGH_NATSIM = 8.3
ENOUGH_SIM = 8.0
ENOUGH_ACCENT = 8.5
ENOUGH_TEXT_HITS = 6
ENOUGH_GENDER = 6
MIN_GAIN = 0.05
LORA_R = 16
LORA_ALPHA = 32.0


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def run(cmd, log_path):
    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n=== {datetime.datetime.now()} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=AUK, env={**os.environ, "PYTHONUTF8": "1"})
    return rc.returncode


def read_train_stats():
    if not os.path.exists(TRAIN_LOG):
        return None
    data = open(TRAIN_LOG, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    upd = re.findall(r"update (\d+)/(\d+)\] loss=([\d.]+)", data)
    vals = re.findall(r"val loss per-t .*?mean=([\d.]+)", data)
    started = re.search(r"TRAIN START ([0-9/: ]+)", data)
    exited = re.search(r"TRAIN EXIT (-?\d+) ([0-9/: ]+)", data)
    return {
        "last_update": int(upd[-1][0]) if upd else 0,
        "total": int(upd[-1][1]) if upd else 0,
        "loss": float(upd[-1][2]) if upd else None,
        "val_means": [float(v) for v in vals],
        "started": started.group(1).strip() if started else None,
        "exited": exited.groups() if exited else None,
    }


def training_pids():
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'auk\\.train\\.train' } | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True)
    return [int(p) for p in ps.stdout.split() if p.strip().isdigit()]


def kill_training():
    for pid in training_pids():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        log(f"killed training pid {pid}")


def gpu1_free():
    try:
        out = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        return int(out.stdout.strip().splitlines()[0]) < 1500
    except Exception:
        return True


def score_results(results):
    scores = []
    for r in results:
        j = r.get("judge") or {}
        o = r.get("objective") or {}
        overall = float(j.get("overall", 0)) / 10
        text_fid = float(j.get("text_fidelity", 0)) / 10
        nat = float(j.get("naturalness", 0)) / 10
        vs = float(j.get("voice_similarity", 0)) / 10
        pal_raw = j.get("palatalization")
        pal = float(pal_raw if pal_raw is not None else j.get("accent", 0)) / 10
        dns = min(float(o.get("gen_dnsmos", 0)), 4.5) / 4.5
        f0g, f0r = float(o.get("gen_f0", 0)), float(o.get("ref_f0", 0))
        f0sim = 1 - min(1.0, abs(math.log2(max(f0g, 60) / max(f0r, 60))) / 1.2)
        s = (0.35 * overall + 0.15 * text_fid + 0.15 * nat + 0.10 * vs
             + 0.10 * dns + 0.10 * pal + 0.05 * f0sim)
        scores.append(round(s, 4))
    return (sum(scores) / max(len(scores), 1)), scores


def judge_means(results):
    js = [r.get("judge") or {} for r in results]
    n = max(len(js), 1)

    def acc(j):
        v = j.get("accent")
        if v is None:
            v = j.get("naturalness", 0)
        return float(v)

    return {
        "overall": sum(float(j.get("overall", 0)) for j in js) / n,
        "naturalness": sum(float(j.get("naturalness", 0)) for j in js) / n,
        "accent": sum(acc(j) for j in js) / n,
        "palatalization": sum(float(j.get("palatalization", acc(j))) for j in js) / n,
        "endings": sum(float(j.get("endings", j.get("naturalness", 0))) for j in js) / n,
        "prosody": sum(float(j.get("prosody", j.get("naturalness", 0))) for j in js) / n,
        "artifacts": sum(float(j.get("artifacts", 10)) for j in js) / n,
        "stress": sum(float(j.get("stress", acc(j))) for j in js) / n,
        "truncated": sum(1 for j in js if j.get("truncated")),
        "wer_mean": sum(float(r.get("wer", 0.0)) for r in results) / n,
        "voice_similarity": sum(float(j.get("voice_similarity", 0)) for j in js) / n,
        "text_matches": sum(1 for j in js if j.get("text_matches")),
        "same_gender": sum(1 for j in js if j.get("same_gender")),
        "native_ok": sum(1 for j in js if j.get("native_ok")),
    }


def deep_eval(update, ckpt_path, label):
    if not gpu1_free():
        log("GPU1 busy, skipping eval this cycle")
        return None
    eval_dir = os.path.join(EVAL_ROOT, f"{label}_u{update}")
    os.makedirs(eval_dir, exist_ok=True)
    if label == "base":
        merged = BASE_CKPT
    else:
        os.makedirs(MERGED_DIR, exist_ok=True)
        merged = os.path.join(MERGED_DIR, f"auk_ru_{update}.safetensors")
        if not os.path.exists(merged) or os.path.getmtime(merged) < os.path.getmtime(ckpt_path):
            log(f"merging adapter update {update}")
            rc = run([PY, os.path.join(LOCAL, "merge_lora.py"), "--run_dir", RUN_DIR,
                      "--ckpt", ckpt_path, "--out", merged,
                      "--base_ckpt", BASE_CKPT, "--lora_r", str(LORA_R), "--lora_alpha", str(LORA_ALPHA)],
                     os.path.join(eval_dir, "merge.log"))
            if rc != 0:
                log(f"merge failed rc={rc}")
                return None
        cfg_dst = os.path.join(MERGED_DIR, "config.yaml")
        if not os.path.exists(cfg_dst):
            shutil.copy(os.path.join(RUN_DIR, "config.yaml"), cfg_dst)
    log(f"deep eval {label} update {update}: generating")
    rc = run([PY, os.path.join(LOCAL, "eval_generate.py"), "--ckpt", merged, "--out_dir", eval_dir],
             os.path.join(eval_dir, "generate.log"))
    if rc != 0:
        log(f"generate failed rc={rc}")
        return None
    log(f"deep eval {label} update {update}: judging")
    results_path = os.path.join(eval_dir, "results.json")
    rc = run([PY, os.path.join(LOCAL, "judge_eval.py"), "--gen_dir", eval_dir, "--out", results_path],
             os.path.join(eval_dir, "judge.log"))
    if rc != 0 or not os.path.exists(results_path):
        log(f"judge failed rc={rc}")
        return None
    results = json.load(open(results_path, encoding="utf-8"))
    score, per = score_results(results)
    means = judge_means(results)
    row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "label": label, "update": update,
           "score": round(score, 4), "per_prompt": per, "means": means, "n_prompts": len(results),
           "results": results_path}
    with open(CONTROL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"EVAL {label} u{update}: score={score:.3f} overall={means['overall']:.1f} accent={means['accent']:.1f} "
        f"palat={means['palatalization']:.1f} nat={means['naturalness']:.1f} "
        f"sim={means['voice_similarity']:.1f} text={means['text_matches']}/{len(results)} "
        f"gender={means['same_gender']}/{len(results)} wer={means['wer_mean']:.2f}")
    prune_merged()
    return row


def prune_merged(keep_updates=(10000, 18000), keep_last_n=2):
    try:
        files = glob.glob(os.path.join(MERGED_DIR, "auk_ru_*.safetensors"))
        up_files = []
        for f in files:
            m = re.search(r"auk_ru_(\d+)\.safetensors", os.path.basename(f))
            if m:
                up_files.append((int(m.group(1)), f))
        up_files.sort(key=lambda x: x[0])
        protected = set(keep_updates)
        for u, _ in up_files[-keep_last_n:]:
            protected.add(u)
        for u, path in up_files:
            if u not in protected:
                try:
                    os.remove(path)
                    log(f"pruned old merged: {os.path.basename(path)}")
                except Exception:
                    pass
    except Exception:
        pass


def finalize(reason, best):
    with open(os.path.join(RUN_DIR, "STOP_REASON.txt"), "w", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()}\n{reason}\n")
    if best and best.get("label") != "base":
        merged = os.path.join(MERGED_DIR, f"auk_ru_{best['update']}.safetensors")
        if os.path.exists(merged):
            dst = os.path.join(RUN_DIR, "auk_ru_best.safetensors")
            shutil.copy(merged, dst)
            with open(os.path.join(RUN_DIR, "BEST.txt"), "w", encoding="utf-8") as f:
                f.write(json.dumps({k: best[k] for k in ("update", "score", "means", "ts")},
                                   ensure_ascii=False, indent=1))
            log(f"finalized: best update {best['update']} score {best['score']} -> {dst}")
    log(f"controller finished: {reason}")


def main():
    global RUN_DIR, EVAL_ROOT, MERGED_DIR, CONTROL_LOG, TRAIN_LOG, ENOUGH_ACCENT, MIN_STOP_UPDATE, MAX_UPDATES
    global BASE_CKPT, LORA_R, LORA_ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--train_log", default=TRAIN_LOG)
    ap.add_argument("--enough_overall", type=float, default=ENOUGH_OVERALL)
    ap.add_argument("--enough_accent", type=float, default=ENOUGH_ACCENT)
    ap.add_argument("--min_stop_update", type=int, default=MIN_STOP_UPDATE)
    ap.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    ap.add_argument("--base_ckpt", default=BASE_CKPT)
    ap.add_argument("--lora_r", type=int, default=LORA_R)
    ap.add_argument("--lora_alpha", type=float, default=LORA_ALPHA)
    args = ap.parse_args()
    RUN_DIR = args.run_dir
    TRAIN_LOG = args.train_log
    ENOUGH_ACCENT = args.enough_accent
    MIN_STOP_UPDATE = args.min_stop_update
    MAX_UPDATES = args.max_updates
    BASE_CKPT = args.base_ckpt
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    EVAL_ROOT = os.path.join(RUN_DIR, "evals")
    MERGED_DIR = os.path.join(RUN_DIR, "merged")
    CONTROL_LOG = os.path.join(RUN_DIR, "control.jsonl")

    os.makedirs(EVAL_ROOT, exist_ok=True)
    started_wall = None
    last_eval_update = -1
    last_eval_time = 0.0
    val_overfit_streak = 0
    history = []
    best = None

    history = []
    if os.path.exists(CONTROL_LOG):
        for line in open(CONTROL_LOG, encoding="utf-8"):
            try:
                history.append(json.loads(line))
            except Exception:
                pass
    log(f"loaded {len(history)} previous eval rows")
    best = max(history, key=lambda r: r["score"]) if history else None
    base_row = next((r for r in history if r.get("label") == "base"), None)
    if base_row is None:
        while base_row is None:
            base_row = deep_eval(0, BASE_CKPT, "base")
            if base_row is None:
                log("baseline eval deferred (GPU busy/errors), retry in 10 min")
                time.sleep(600)
        history.append(base_row)
    if best is None or base_row["score"] > best["score"]:
        best = base_row
    last_eval_update = max((r["update"] for r in history), default=0)

    while True:
        try:
            stats = read_train_stats()
            alive = bool(training_pids())
            if stats is None:
                time.sleep(120)
                continue

            if stats["started"] and started_wall is None:
                try:
                    started_wall = datetime.datetime.strptime(stats["started"], "%m/%d/%Y %H:%M:%S")
                except Exception:
                    started_wall = datetime.datetime.now()
            now = time.time()
            new_ckpt = os.path.exists(os.path.join(RUN_DIR, "model_last.pt"))
            upd_delta = stats["last_update"] - last_eval_update
            due = (new_ckpt and upd_delta >= MIN_UPDATES_BETWEEN_EVALS
                   and (now - last_eval_time) >= MIN_MINUTES_BETWEEN_EVALS * 60)

            if alive and due:
                row = deep_eval(stats["last_update"], os.path.join(RUN_DIR, "model_last.pt"), "lora")
                if row:
                    last_eval_update = row["update"]
                    last_eval_time = now
                    history.append(row)
                    if best is None or row["score"] > best["score"]:
                        best = row
                        log(f"new best: u{row['update']} score {row['score']}")

                    stop_reason = None
                    m = row["means"]
                    lora_evals = [h for h in history if h["label"] == "lora"]
                    recent = lora_evals[-2:]
                    recent_med = (sorted(r["score"] for r in recent)[len(recent) // 2]
                                  if len(recent) >= 2 else row["score"])
                    enough = (stats["last_update"] >= MIN_STOP_UPDATE
                              and m["overall"] >= ENOUGH_OVERALL
                              and m["naturalness"] >= ENOUGH_NATSIM
                              and m["voice_similarity"] >= ENOUGH_SIM
                              and m["accent"] >= ENOUGH_ACCENT
                              and m.get("palatalization", m["accent"]) >= ENOUGH_ACCENT
                              and m["text_matches"] >= ENOUGH_TEXT_HITS
                              and m["same_gender"] >= ENOUGH_GENDER
                              and base_row is not None
                              and (row["score"] - base_row["score"]) >= MIN_GAIN)
                    improving = best is not None and (best["score"] - recent_med) < 0.01
                    if enough and improving and len(lora_evals) >= 3:
                        stop_reason = (f"enough quality: overall={m['overall']:.1f} accent={m['accent']:.1f} "
                                       f"nat={m['naturalness']:.1f} sim={m['voice_similarity']:.1f} "
                                       f"text={m['text_matches']}/6 gain={row['score']-base_row['score']:+.3f}")
                    if len(recent) == 2 and recent_med < best["score"] - 0.08:
                        stop_reason = (f"degradation: median of last two {recent_med:.3f} < best {best['score']:.3f}")
                    if len(recent) == 2 and all(r["means"].get("wer_mean", 0.0) >= 0.25 for r in recent):
                        stop_reason = ("ASR errors: two consecutive evals with word error rate >= 0.25 "
                                       f"({recent[0]['means'].get('wer_mean')}, {recent[1]['means'].get('wer_mean')})")
                    vals = stats["val_means"]
                    if len(vals) >= 2:
                        best_val = min(vals)
                        if vals[-1] > best_val * 1.02 and vals[-1] >= vals[-2]:
                            val_overfit_streak += 1
                        else:
                            val_overfit_streak = 0
                        if val_overfit_streak >= 2:
                            stop_reason = (f"val overfit: last={vals[-1]:.4f} vs best={best_val:.4f} "
                                           f"(streak {val_overfit_streak})")
                    if len(vals) >= 3 and vals[-1] > vals[-2] > vals[-3] and len(recent) == 2 and recent[-1]["score"] <= recent[0]["score"]:
                        stop_reason = f"val loss rising: {vals[-3:]}, evals not improving"
                    if stats["last_update"] >= MAX_UPDATES:
                        stop_reason = f"update cap {MAX_UPDATES} reached"
                    if started_wall and (datetime.datetime.now() - started_wall).total_seconds() > MAX_HOURS * 3600:
                        stop_reason = f"wall-clock cap {MAX_HOURS}h reached"

                    if stop_reason:
                        log(f"STOP: {stop_reason}")
                        kill_training()
                        finalize(stop_reason, best)
                        return

            if not alive:
                if stats["last_update"] > last_eval_update and new_ckpt:
                    row = deep_eval(stats["last_update"], os.path.join(RUN_DIR, "model_last.pt"), "lora")
                    if row:
                        history.append(row)
                        if best is None or row["score"] > best["score"]:
                            best = row
                finalize("training process exited", best)
                return

            time.sleep(120)
        except Exception as exc:
            log(f"controller loop error: {type(exc).__name__}: {exc}")
            time.sleep(120)


if __name__ == "__main__":
    main()
