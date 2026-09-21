# -*- coding: utf-8 -*-
"""Упаковка тренировочного чекпойнта в ЧИСТЫЙ релизный LoRA-адаптер.

Убирает optimizer_state_dict / scheduler_state_dict (нужны только для resume,
в релизе — лишний вес: 615 МБ → ~250 МБ) и добавляет метаданные релиза.

Содержимое: peft=True, model_state_dict (400 LoRA-тензоров), extra_state_dict (18),
update, lora_r, lora_alpha + release_* поля.

Проверка целостности: число тензоров и суммарный размер совпадают с оригиналом.

usage: python make_release_adapter.py --src <model_N.pt> --out <release.pt> --name "AuK-ru v1.0"
"""
import argparse
import hashlib
import json
import os

import torch


def count_and_size(state):
    n = len(state)
    tot = sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))
    return n, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="AuK-ru")
    ap.add_argument("--stage", default="")
    ap.add_argument("--sha256_json", default="")
    args = ap.parse_args()

    s = torch.load(args.src, map_location="cpu", weights_only=False)
    assert s.get("peft") is True, "это не peft-чекпойнт"

    msd, esd = s["model_state_dict"], s.get("extra_state_dict") or {}
    n0, sz0 = count_and_size(msd)
    ne0, sze0 = count_and_size(esd)

    rel = {
        "peft": True,
        "model_state_dict": msd,
        "extra_state_dict": esd,
        "update": s.get("update"),
        "lora_r": s.get("lora_r"),
        "lora_alpha": s.get("lora_alpha"),
        "release_name": args.name,
        "release_stage": args.stage or os.path.basename(args.src),
        "release_base": "auk_ru_10000.safetensors (sha256 a9ac0b81f5159ce707afc31e4897be2236716163144447092f7a0e0908642a24)",
        "release_note": "LoRA adapter only (optimizer/scheduler state removed). Merge with merge_lora.py.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(rel, args.out)

    # верификация
    back = torch.load(args.out, map_location="cpu", weights_only=False)
    n1, sz1 = count_and_size(back["model_state_dict"])
    ne1, sze1 = count_and_size(back["extra_state_dict"])
    ok = (n1 == n0 and sz1 == sz0 and ne1 == ne0 and sze1 == sze0
          and back["update"] == s.get("update") and back["lora_r"] == s.get("lora_r"))
    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)

    print(f"src : {n0} lora tensors ({sz0/1e6:.1f} MB) + {ne0} extra ({sze0/1e6:.1f} MB)")
    print(f"out : {n1} lora tensors ({sz1/1e6:.1f} MB) + {ne1} extra ({sze1/1e6:.1f} MB)")
    print(f"file: {os.path.getsize(args.out)/1e6:.1f} MB (src {os.path.getsize(args.src)/1e6:.1f} MB)")
    print(f"sha256: {h.hexdigest()}")
    print(f"integrity: {'PASS' if ok else 'FAIL'}")
    assert ok

    if args.sha256_json:
        d = json.load(open(args.sha256_json, encoding="utf-8")) if os.path.exists(args.sha256_json) else {}
        d[os.path.basename(args.out)] = {"sha256": h.hexdigest(), "size": os.path.getsize(args.out),
                                         "update": s.get("update"), "lora_r": s.get("lora_r"),
                                         "lora_alpha": s.get("lora_alpha")}
        json.dump(d, open(args.sha256_json, "w", encoding="utf-8"), indent=1)
        print("registered in", args.sha256_json)


if __name__ == "__main__":
    main()
