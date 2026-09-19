"""Dev-набор v2 (шаг 3): продуктовый путь run_generate (auto-retry: tight/template3,
WER-выбор, trim, limit_peak). RAW-версия остаётся в dev_set/u18000 как исторический срез.
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

import auk.infer.infer_gradio as ig  # noqa: E402
from auk.infer.infer_auk import AukInfer  # noqa: E402
from auk.infer import quality  # noqa: E402
from dev_set import TEXTS, VOICES  # noqa: E402

OUT = os.path.join(AUK, "local_tests", "dev_set", "u18000_prod")
MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")

ig.CKPT_PATHS["dev"] = MERGED
ig.ENGINES["dev"] = AukInfer(
    config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
    ckpt_path=MERGED,
    qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
    cpu_offload=True,
    device="cuda:0",
    dtype="bf16",
)


def main():
    import torch
    import soundfile as sf

    os.makedirs(OUT, exist_ok=True)
    voice_names = list(VOICES.keys())
    manifest = []
    for i, (cat, text) in enumerate(TEXTS):
        vname = voice_names[i % len(voice_names)]
        ref = VOICES[vname]
        sr, pcm = ig.run_generate("dev", ref, text, None, "", "", 64, 2.0, 1234,
                                  translit=False, accentize=True, bestofn=1, trim=True, tts_wrap=True)
        out = os.path.join(OUT, f"devp{i:02d}_{cat}_{vname}.wav")
        sf.write(out, pcm, sr, subtype="PCM_16")
        x = pcm.astype("float32") / 32768.0
        heard = quality.transcribe(torch.from_numpy(x).unsqueeze(0), sr)
        m = quality.wer_metrics(text, heard)
        manifest.append({"idx": i, "cat": cat, "text": text, "voice": vname, "ref": ref,
                         "file": out, "mode": "product", "seed": 1234,
                         "wer": round(m["wer"], 3), "ins": m["ins"], "heard": heard})
        print(f"devp{i:02d} {cat:<10} {vname:<11} wer={m['wer']:.2f} ins={m['ins']}", flush=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("DEV_SET_PROD_DONE")


if __name__ == "__main__":
    main()
