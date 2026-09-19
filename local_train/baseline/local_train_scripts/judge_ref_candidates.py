"""Judge reference-voice candidates: gender + naturalness via Qwen2.5-Omni."""
import json
import os
import re

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

AUK = r"G:\AI\AuK"
QWEN = os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B")
CANDS = os.path.join(AUK, "local_train", "female_ref_candidates.json")

PROMPT = """Listen to this audio clip. It should be a natural human FEMALE voice speaking Russian.
Answer with ONLY a JSON object:
{"female": true/false, "naturalness": 0-10, "clarity": 0-10, "noise_or_artifacts": true/false, "comment": "short"}"""


def parse(text):
    m = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for c in reversed(m):
        try:
            o = json.loads(c)
            if "female" in o:
                return o
        except Exception:
            continue
    return None


def main():
    cands = json.load(open(CANDS, encoding="utf-8"))[:6]
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, device_map="cuda:1")
    model.eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(QWEN)
    out = []
    for c in cands:
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio": c["path"]},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = proc(text=prompt, audio=audios, images=images, videos=videos,
                      return_tensors="pt", padding=True, use_audio_in_video=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=128)
        reply = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        j = parse(reply)
        rec = dict(c)
        rec["judge"] = j
        out.append(rec)
        print("f0=%.0f | %s | %s | %s" % (c["metrics"]["f0"], os.path.basename(c["path"]), j, c["text"][:40]), flush=True)
    json.dump(out, open(os.path.join(AUK, "local_train", "female_ref_judged.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("REFJUDGE_DONE")


if __name__ == "__main__":
    main()
