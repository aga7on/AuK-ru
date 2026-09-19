"""Compute FULL sha256 for s2 merges and every real upstream dependency (one-shot, resumable)."""
import hashlib
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_sha256.json")
AUK = r"G:\AI\AuK"
TARGETS = {
    "upstream_base": r"G:\AI\AuK\ckpts\AuK\auk_base.safetensors",
    "s0_merged_best": r"G:\AI\AuK\local_train\run_ru\auk_ru_best.safetensors",
    "s1_merged_10000": r"G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors",
    "s1_adapter_10000": r"G:\AI\AuK\local_train\run_ru_s1\model_10000.pt",
    "s2_A250": r"G:\AI\AuK\local_train\run_s2_A\merged\auk_s2_A_250.safetensors",
    "s2_A500": r"G:\AI\AuK\local_train\run_s2_A\merged\auk_s2_A_500.safetensors",
    "s2_B250": r"G:\AI\AuK\local_train\run_s2_B\merged\auk_s2_B_250.safetensors",
    "s2_B500": r"G:\AI\AuK\local_train\run_s2_B\merged\auk_s2_B_500.safetensors",
    "s2_A_adapter_250": r"G:\AI\AuK\local_train\run_s2_A\model_250.pt",
    "s2_A_adapter_500": r"G:\AI\AuK\local_train\run_s2_A\model_500.pt",
    "s2_B_adapter_250": r"G:\AI\AuK\local_train\run_s2_B\model_250.pt",
    "s2_B_adapter_500": r"G:\AI\AuK\local_train\run_s2_B\model_500.pt",
}


def full_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT, encoding="utf-8"))
    for name, p in TARGETS.items():
        if name in res and res[name].get("sha256"):
            continue
        if not os.path.exists(p):
            res[name] = {"path": p, "sha256": None, "error": "missing", "size": None}
        else:
            res[name] = {"path": p, "sha256": full_sha(p), "size": os.path.getsize(p)}
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(name, res[name]["sha256"], flush=True)
    print("HASH_DONE", flush=True)


if __name__ == "__main__":
    main()
