"""Objective re-check of s2 artifacts (16.09.2026). Read-only, prints JSON."""
import hashlib
import json
import os

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")
MAIN = ["u0_control", "s2_A_250", "s2_A_500", "s2_B_250", "s2_B_500"]
PHON = ["phon_u10000", "phon_A_250", "phon_A_500", "phon_B_250", "phon_B_500"]


def sha16(path, limit_mb=2048):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    left = limit_mb * 1024 * 1024
    with open(path, "rb") as f:
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()[:16]


def check_dir(name):
    d = os.path.join(LT, name)
    out = {"dir": d, "exists": os.path.isdir(d)}
    man_path = os.path.join(d, "manifest.json")
    out["manifest"] = os.path.exists(man_path)
    if out["manifest"]:
        man = json.load(open(man_path, encoding="utf-8"))
        out["manifest_n"] = len(man)
        ids = [m["id"] for m in man]
        out["manifest_uniq_ids"] = len(set(ids))
    rp = os.path.join(d, "run_params.json")
    out["run_params"] = os.path.exists(rp)
    if out["run_params"]:
        r = json.load(open(rp, encoding="utf-8"))
        out["run_status"] = r.get("status")
        out["run_ckpt"] = r.get("ckpt")
        out["run_ckpt_exists"] = os.path.exists(r.get("ckpt", ""))
        out["run_ckpt_sha16_claimed"] = r.get("ckpt_sha16")
        out["run_ckpt_sha16_actual"] = sha16(r.get("ckpt", "")) if r.get("ckpt") else None
    wavs = [f for f in os.listdir(d) if f.endswith(".wav")] if os.path.isdir(d) else []
    out["wav_n"] = len(wavs)
    if out["manifest"]:
        expected = set(os.path.basename(m["file"]) for m in man)
        got = set(wavs)
        out["missing_wav"] = sorted(expected - got)[:10]
        out["extra_wav"] = sorted(got - expected)[:10]
    return out


def main():
    res = {"main": [check_dir(x) for x in MAIN], "phon": [check_dir(x) for x in PHON]}

    blind = os.path.join(LT, "blind_s2")
    import csv
    listen = list(csv.DictReader(open(os.path.join(blind, "LISTEN.csv"), encoding="utf-8")))
    secret = list(csv.DictReader(open(os.path.join(blind, "SECRET_map.csv"), encoding="utf-8")))
    form = list(csv.DictReader(open(os.path.join(blind, "listen_form.csv"), encoding="utf-8")))
    by_group, by_variant = {}, {}
    for r in listen:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    for r in secret:
        by_variant[r["variant"]] = by_variant.get(r["variant"], 0) + 1
    res["blind"] = {"files": len(listen), "wav_n": len([f for f in os.listdir(os.path.join(blind, "wav")) if f.endswith(".wav")]),
                    "by_group": by_group, "by_variant": by_variant,
                    "form_filled": sum(1 for r in form if any(str(r[k]).strip() for k in
                                       ("text_accuracy", "pronunciation", "naturalness", "voice_similarity", "notes")))}

    merged = {}
    for run, step in [("A", "250"), ("A", "500"), ("B", "250"), ("B", "500")]:
        p = os.path.join(AUK, "local_train", f"run_s2_{run}", "merged", f"auk_s2_{run}_{step}.safetensors")
        merged[f"{run}@{step}"] = {"path": p, "exists": os.path.exists(p), "sha16": sha16(p)}
    res["merged_after_remerge"] = merged
    res["adapters"] = {f"{r}_{s}": os.path.exists(os.path.join(AUK, "local_train", f"run_s2_{r}", f"model_{s}.pt"))
                       for r in "AB" for s in ("250", "500")}
    res["sha16_definition"] = "sha256 of first 2048 MB only (verify_eval_run.py::sha16)"
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
