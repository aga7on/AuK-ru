# -*- coding: utf-8 -*-
"""S15: композиция LoRA-адаптеров поверх одной базы (core + Σ αᵢ·ΔWᵢ).

ΔWᵢ = merge(base, adapterᵢ) − base; результат: W = base + Σ αᵢ·ΔWᵢ.
Корректно для всех наших стадий: единая база auk_ru_10000, r32/α64 (LINEAGE).

Паритет-инвариант: compose(base, [adapter], [1.0]) должен совпадать с merge_lora.py
побайтово (проверяется --parity против существующего merged-файла).

usage:
  python compose_adapters.py --adapter run_s7\model_5750.pt:1.0 --adapter run_s5\model_4500.pt:0.5 \
      --out composed.safetensors [--parity run_s7\merged\auk_s7_5750.safetensors]
"""
import argparse
import os
import sys

import torch

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

from merge_lora import build_model  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

BASE = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_10000.safetensors")
CONFIG = os.path.join(AUK, "local_train", "run_s2_B", "merged", "config.yaml")
LORA_TARGETS = ["to_qkv", "to_qkv_c", "to_out.0", "to_out_c",
                "ff.linear_in", "ff.linear_out",
                "ff_c.linear_in", "ff_c.linear_out",
                "ff_x.linear_in", "ff_x.linear_out",
                "attn_norm.linear", "attn_norm_x.linear", "attn_norm_c.linear"]


def load_base_model():
    model_config = OmegaConf.load(CONFIG).model
    latent_dim = OmegaConf.load(CONFIG).model.vae.latent_dim
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    schedule = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    base = load_file(BASE, device="cpu")
    num_layers = int(base["layer_weights"].shape[0])
    model = build_model(arch, latent_dim, schedule, num_layers)
    model.load_state_dict(base, strict=False)
    return model, base


def adapter_delta(model, base, ckpt_path):
    """ΔW = merge(base, adapter) − base (состояние модели восстанавливается из base)."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    # восстановить чистую базу
    model.load_state_dict(base, strict=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    r = int(state.get("lora_r", 32))
    alpha = float(state.get("lora_alpha", 64.0))
    m = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                                         bias="none", target_modules=LORA_TARGETS))
    set_peft_model_state_dict(m, state["model_state_dict"])
    if state.get("extra_state_dict"):
        m.load_state_dict(state["extra_state_dict"], strict=False)
    m = m.merge_and_unload()
    merged = {k: v for k, v in m.state_dict().items() if not k.startswith("text_encoder.")}
    delta = {}
    for k, v in merged.items():
        if k in base and v.dtype in (torch.float32, torch.bfloat16, torch.float16):
            delta[k] = (v.float() - base[k].float())
        # не-тензорные/отсутствующие в базе ключи (layer_weights и пр.) — берём как есть при α=1
    return merged, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True,
                    help="path/to/model_N.pt:weight (weight — коэффициент α)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity", default=None, help="существующий merged для побайтовой сверки (одиночный α=1)")
    args = ap.parse_args()

    model, base = load_base_model()
    out_state = {k: v.clone() for k, v in base.items()}

    for spec in args.adapter:
        path, w = spec.rsplit(":", 1)
        weight = float(w)
        path = path if os.path.isabs(path) else os.path.join(AUK, path)
        merged, delta = adapter_delta(model, base, path)
        for k, d in delta.items():
            out_state[k] = out_state[k].float() + weight * d
        # ключи вне delta (если адаптер их менял) — при одиночной композиции берём merged
        if len(args.adapter) == 1 and weight == 1.0:
            for k, v in merged.items():
                if k not in delta:
                    out_state[k] = v
        print(f"adapter {os.path.basename(path)} α={weight} applied ({len(delta)} tensors)")

    # dtype как у базы
    for k in out_state:
        if k in base:
            out_state[k] = out_state[k].to(base[k].dtype)

    if args.parity:
        ref = load_file(args.parity, device="cpu")
        bad, maxdiff = 0, 0.0
        for k, v in ref.items():
            if k not in out_state:
                bad += 1
                continue
            d = (out_state[k].float() - v.float()).abs().max().item()
            maxdiff = max(maxdiff, d)
            if d > 1e-4:
                bad += 1
        print(f"PARITY vs {os.path.basename(args.parity)}: mismatched={bad}/{len(ref)} maxdiff={maxdiff:.2e}")
        if bad == 0:
            print("PARITY PASS: compose(α=1) == merge_lora побайтово (в пределах 1e-4)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out_state, args.out)
    print(f"saved {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
