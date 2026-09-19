"""Read-only asset probe for the toolkit inventory worker.

Collects duration / sample-rate / channels / frame-count / sha256 for candidate
heldout fixtures (upstream demo assets, local control refs) and size+sha256-prefix
for the two model controls (upstream base, s1 u10000 merged). Writes assets_probe.json
next to this script. No audio is generated or modified.
"""
from __future__ import annotations

import hashlib
import json
import os

import soundfile as sf

AUK = r"G:\AI\AuK"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_probe.json")

WAV_GLOBS = [
    ("upstream_demo", os.path.join(AUK, "assets", "demo-input-audio")),
    ("local_after_pe", os.path.join(AUK, "assets", "after_pe")),
    ("local_controls", os.path.join(AUK, "local_tests")),
]

CKPTS = [
    ("upstream_base", os.path.join(AUK, "ckpts", "AuK", "auk_base.safetensors")),
    ("s1_u10000_merged", os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_10000.safetensors")),
]


def sha256(path: str, limit: int | None = None) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if limit is not None and n >= limit:
                break
    return h.hexdigest(), n


def wav_info(path: str) -> dict:
    info = sf.info(path)
    rec = {
        "path": path,
        "sr": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": round(info.frames / info.samplerate, 3) if info.samplerate else None,
        "subtype": info.subtype,
        "size_bytes": os.path.getsize(path),
    }
    h, _ = sha256(path)
    rec["sha256"] = h
    return rec


def main() -> None:
    out: dict = {"wavs": {}, "ckpts": {}}
    for group, root in WAV_GLOBS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.lower().endswith(".wav"):
                    continue
                p = os.path.join(dirpath, name)
                key = os.path.relpath(p, AUK).replace("\\", "/")
                try:
                    out["wavs"][key] = {"group": group, **wav_info(p)}
                except Exception as exc:  # noqa: BLE001
                    out["wavs"][key] = {"group": group, "path": p, "error": f"{type(exc).__name__}: {exc}"}
    for label, path in CKPTS:
        if os.path.isfile(path):
            h, n = sha256(path, limit=2048 * 1024 * 1024)
            out["ckpts"][os.path.relpath(path, AUK).replace("\\", "/")] = {
                "label": label,
                "size_bytes": os.path.getsize(path),
                "sha256_first_2GiB": h,
                "bytes_hashed": n,
            }
        else:
            out["ckpts"][os.path.relpath(path, AUK).replace("\\", "/")] = {"label": label, "missing": True}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {len(out['wavs'])} wavs, {len(out['ckpts'])} ckpts")


if __name__ == "__main__":
    main()
