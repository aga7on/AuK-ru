"""Judge specific generated samples against their ground-truth target audio (tgt as reference)."""
import json
import os
import sys

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))

from judge_eval import TEMPLATE, dnsmos_ovrl, f0_median, parse_json, read_mono  # noqa: E402

SAMPLES = [
    ("u5000_top", os.path.join(AUK, "local_tests", "keeper", "u5000_top_gen.wav"),
     os.path.join(AUK, "local_tests", "keeper", "u5000_tgt.wav")),
    ("u6500_bad", os.path.join(LOCAL, "run_ru_s1", "samples", "update_6500_gen.wav"),
     os.path.join(LOCAL, "run_ru_s1", "samples", "update_6500_tgt.wav")),
]


def main():
    from speechmos import dnsmos as dnsmos_mod

    row = json.loads(open(os.path.join(LOCAL, "data", "val.jsonl"), encoding="utf-8").readline())
    text = row["messages"][0]["content"][0]["text"]
    body = text.split("'", 2)[1] if "'" in text else text
    print("intended text:", body[:120].encode("ascii", "replace").decode(), flush=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"), torch_dtype=torch.bfloat16, device_map="cuda:1")
    model.eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"))

    for label, gen, ref in SAMPLES:
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio": ref},
            {"type": "audio", "audio": gen},
            {"type": "text", "text": TEMPLATE.format(text=body)},
        ]}]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = proc(text=prompt, audio=audios, images=images, videos=videos,
                      return_tensors="pt", padding=True, use_audio_in_video=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=192)
        reply = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        j = parse_json(reply)
        gx, gsr = read_mono(gen)
        obj = {"dur": round(len(gx) / gsr, 2), "f0": round(f0_median(gx, gsr), 1),
               "dnsmos": round(dnsmos_ovrl(gx, gsr, dnsmos_mod), 2)}
        print(f"== {label}: judge={j} obj={obj}", flush=True)


if __name__ == "__main__":
    main()
