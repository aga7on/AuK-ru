r"""Чистый u10000: контрольный сэмпл теми же текстом/рефом/параметрами, что обучающие
сэмплы run_s2_A\samples\update_*_gen.wav (точная реплика synthesize_and_save из train.py).

usage: gen_u10000_sample.py [--ckpt <merged.safetensors>] [--out <dir>] [--device cuda:0]
"""
import argparse
import copy
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))

import torch
import torchaudio
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"G:\AI\AuK\local_train\run_ru_s1\merged\auk_ru_10000.safetensors")
    ap.add_argument("--out", default=r"G:\AI\AuK\local_tests\phonetic_control")
    ap.add_argument("--config", default=r"G:\AI\AuK\local_train\s2_config.yaml")
    ap.add_argument("--val", default=r"G:\AI\AuK\local_train\data_s2_full\v2_after_identity\val.jsonl")
    ap.add_argument("--ref", default=r"G:\AI\AuK\local_tests\user_ref.wav")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="u10000")
    args = ap.parse_args()

    from auk.train.train import _apply_pron_lexicon

    device = args.device
    cfg = OmegaConf.load(args.config).model
    text_cfg = cfg.text_encoder
    vae_cfg = cfg.vae

    row = json.loads(open(args.val, encoding="utf-8").readline())
    conv = copy.deepcopy([m for m in row["messages"] if m["role"] != "assistant"])
    for msg in conv:
        if msg.get("role") != "user":
            continue
        for item in msg["content"]:
            if item["type"] == "text":
                item["text"] = item["text"].replace("|<no_prompt_audio>|", "")
                item["text"] = _apply_pron_lexicon(item["text"])
        msg["content"].append({"type": "audio", "audio": args.ref})

    target_path = None
    for m in row["messages"]:
        if m["role"] == "assistant":
            for c in m["content"]:
                if c["type"] == "audio":
                    target_path = c.get("audio") or c.get("audio_url")
    assert target_path and os.path.exists(target_path), f"target not found: {target_path}"

    print("loading Qwen thinker ...", flush=True)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        text_cfg.text_encoder_path, torch_dtype=torch.bfloat16)
    if thinker.visual is not None:
        del thinker.visual
        thinker.visual = None
    processor = Qwen2_5OmniProcessor.from_pretrained(text_cfg.text_encoder_path)

    print("loading VAE ...", flush=True)
    mk = OmegaConf.to_container(vae_cfg.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(vae_name=vae_cfg.vae_name, vae_cfg=BigVGANFlowVAEConfig.from_dict(mk),
                               vae_ckpt=vae_cfg.vae_model_path, map_location="cpu")

    print("building CFMEdit ...", flush=True)
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    arch["attn_backend"] = "torch"
    sched = OmegaConf.to_container(cfg.get("schedule", OmegaConf.create({})), resolve=True)
    model = CFMEdit(transformer=Flux2Edit(**arch, latent_dim=vae_cfg.latent_dim),
                    text_encoder=thinker, text_processor=processor,
                    num_channels=vae_cfg.latent_dim, **sched)

    print(f"loading weights {args.ckpt} ...", flush=True)
    sd = load_file(args.ckpt, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} (text_encoder.*={sum(1 for k in missing if k.startswith('text_encoder.'))}) "
          f"unexpected={len(unexpected)}", flush=True)

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    vae_model.requires_grad_(False)

    def load24(path):
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        if sr != 24000:
            w = torchaudio.functional.resample(w, sr, 24000)
        return w.unsqueeze(0).to(device)

    ref_audio = load24(args.ref)
    tgt_audio = load24(target_path)
    ref_lens = torch.tensor([ref_audio.size(-1)], device=device, dtype=torch.long)
    tgt_lens = torch.tensor([tgt_audio.size(-1)], device=device, dtype=torch.long)

    with torch.no_grad():
        ref_latent, ref_l = vae_model.encoding_and_normalization(ref_audio, sample_lengths=ref_lens)
        tgt_latent, tgt_l = vae_model.encoding_and_normalization(tgt_audio, sample_lengths=tgt_lens)
        ref_len = ref_l[0].item()
        tgt_len = tgt_l[0].item()
        cond_inputs = model.build_cond_inputs([conv], model.text_processor)
        # текст-энкодер нужен лишь один раз (в начале sample); освобождаем GPU перед ODE-циклом
        text_embeds, context_mask = model.encode_text(cond_inputs, device)
        model.text_encoder = model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        model.encode_text = lambda _text, _dev: (text_embeds, context_mask)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated, _ = model.sample(cond=ref_latent, text=None,
                                        duration=ref_len + tgt_len, lens=ref_l,
                                        steps=64, cfg_strength=2.0, sway_sampling_coef=-1.0,
                                        seed=1234)
        generated = generated.to(torch.float32)

    os.makedirs(args.out, exist_ok=True)
    for name, latent in [("gen", generated[:, ref_len:ref_len + tgt_len, :]),
                         ("tgt", tgt_latent[:1, :tgt_len, :])]:
        lat = vae_model.denormalize(latent.to(device))
        audio = vae_model.inference_from_latents(lat.permute(0, 2, 1)).cpu().to(torch.float32)
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        out = os.path.join(args.out, f"update_{args.tag}_{name}.wav")
        torchaudio.save(out, audio, 24000)
        print(f"saved {out} ({audio.shape[-1] / 24000:.2f}s)", flush=True)

    json.dump({"ckpt": args.ckpt, "ref": args.ref, "target": target_path,
               "text": conv[0]["content"][0]["text"], "steps": 64, "cfg": 2.0, "sway": -1.0,
               "seed": 1234, "duration_frames": ref_len + tgt_len},
              open(os.path.join(args.out, f"update_{args.tag}_params.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("U10000_SAMPLE_DONE")


if __name__ == "__main__":
    main()
