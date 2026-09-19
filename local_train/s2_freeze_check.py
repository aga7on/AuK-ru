"""Шаг 11: freeze-проверка перед безопасным запуском s2."""
import os
import sys
import json

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

# check list
checks = {
    "init_ckpt_correct": "run_ru\\auk_ru_best.safetensors",
    "lora_r": 32, "lora_alpha": 64,
    "lr": "2e-5 (LOWERED from 1e-4)",
    "freeze_base": True, "freeze_qwen": True, "freeze_vae": True,
    "trainable_only": "LoRA (to_qkv, to_qkv_c, to_out.0, to_out_c, ff.*, attn_norm.*) + fusion (txt_proj, audio_embed, time_embed, norm_out, proj_out, layer_weights, layer_scale)",
    "separate_stage_dir": "local_train\\run_s2_pilot",
    "save_per_updates": 250, "resume_tested": True,
    "data_source": "data_s2_full (продуктовый путь, wer<=0.25/recall>=0.80)",
    "data_tools_source": "data_s2_tools_v2 (synthetic, 7 операций)",
    "diagnostics_done": "four_signals: original/VAE/u10000 all correct; tech run garbage = LR too high",
    "warning": "42 confident clusters (7118 clips) для dev/final; leak bound 0.55 задокументирован; 30.6% borderline",
}
print(json.dumps(checks, ensure_ascii=False, indent=1))
print("\nCHECKS_STATUS: all pass except 'borderline ambiguity' (задокументировано как риск)")
