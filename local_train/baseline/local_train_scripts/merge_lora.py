"""Merge a LoRA adapter checkpoint produced by auk.train.train into a full safetensors file.

Usage:
  python merge_lora.py --ckpt local_train/run_ru/model_last.pt
                      --out  local_train/run_ru/merged/auk_ru_UPDATE.safetensors
                      --config ckpts/AuK/config.yaml
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

AUK_ROOT = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK_ROOT, "src"))

from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file


class _DummyTE(torch.nn.Module):
    """Stand-in for the frozen Qwen encoder: only config depth is needed to build layer_weights."""

    def __init__(self, num_layers):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_layers))


def build_model(arch, latent_dim, schedule, num_layers):
    from auk.model.cfm_edit import CFMEdit
    from auk.model.flux2_edit import Flux2Edit

    arch = dict(arch)
    arch["attn_backend"] = "torch"
    return CFMEdit(
        transformer=Flux2Edit(**arch, latent_dim=latent_dim),
        text_encoder=_DummyTE(num_layers),
        text_processor=None,
        num_channels=latent_dim,
        **schedule,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="training output dir (for config.yaml)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_ckpt", default=os.path.join(AUK_ROOT, "ckpts", "AuK", "auk_base.safetensors"))
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    args = ap.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    model_config = OmegaConf.load(config_path).model
    latent_dim = OmegaConf.load(config_path).model.vae.latent_dim
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    schedule = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    update = int(state.get("update", -1))
    lora_r = int(state.get("lora_r", args.lora_r))
    lora_alpha = float(state.get("lora_alpha", args.lora_alpha))
    print(f"checkpoint update={update} peft={bool(state.get('peft'))} r={lora_r} alpha={lora_alpha}")

    base = load_file(args.base_ckpt, device="cpu")
    num_layers = int(base["layer_weights"].shape[0])
    model = build_model(arch, latent_dim, schedule, num_layers)

    missing, unexpected = model.load_state_dict(base, strict=False)
    non_te_missing = [k for k in missing if not k.startswith("text_encoder")]
    print(f"base loaded ({os.path.basename(args.base_ckpt)}) | missing(non-TE)={len(non_te_missing)} unexpected={len(unexpected)}")

    if state.get("peft"):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        lora_targets = ["to_qkv", "to_qkv_c", "to_out.0", "to_out_c",
                        "ff.linear_in", "ff.linear_out",
                        "ff_c.linear_in", "ff_c.linear_out",
                        "ff_x.linear_in", "ff_x.linear_out",
                        "attn_norm.linear", "attn_norm_x.linear", "attn_norm_c.linear"]
        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                                 bias="none", target_modules=lora_targets))
        set_peft_model_state_dict(model, state["model_state_dict"])
        if state.get("extra_state_dict"):
            model.load_state_dict(state["extra_state_dict"], strict=False)
        model = model.merge_and_unload()
        print("adapter merged")
    else:
        model.load_state_dict(state["model_state_dict"], strict=False)

    out_state = {k: v for k, v in model.state_dict().items() if not k.startswith("text_encoder.")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out_state, args.out)
    print(f"saved {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")
    return update


if __name__ == "__main__":
    main()
