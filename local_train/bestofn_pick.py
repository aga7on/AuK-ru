"""Best-of-N step 2: transcribe candidates (Qwen ASR), score vs text + pauses + DNSMOS, pick best."""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

from judge_eval import (ASR_PROMPT, QWEN_PATH, dnsmos_ovrl, read_mono,  # noqa: E402
                        recall_and_missing)
from pause_stats import rms_frames  # noqa: E402


def silence_profile(x, sr):
    r = rms_frames(x)
    db = 20 * np.log10(r + 1e-9)
    speech = db > (db.max() - 35)
    hop_s = 240 / sr
    lead = 0.0
    for i, s in enumerate(speech):
        if s:
            lead = i * hop_s
            break
    tail = 0.0
    for i in range(len(speech) - 1, -1, -1):
        if speech[i]:
            tail = (len(speech) - 1 - i) * hop_s
            break
    excess = max(0.0, lead - 0.3) + max(0.0, tail - 0.3)
    runs = []
    start = None
    for i, s in enumerate(speech):
        if not s and start is None:
            start = i
        elif s and start is not None:
            if (i - start) * hop_s >= 0.15:
                runs.append(((start * hop_s), (i - start) * hop_s))
            start = None
    for p, d in runs:
        if p > lead + 0.2 and p < len(x) / sr - tail - 0.2:
            excess += max(0.0, d - 0.45)
    return lead, tail, excess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from speechmos import dnsmos as dnsmos_mod

    manifest = json.load(open(os.path.join(args.dir, "manifest.json"), encoding="utf-8"))
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_PATH)

    rows = []
    for item in manifest:
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio": item["gen"]},
            {"type": "text", "text": ASR_PROMPT},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(text=prompt, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=128)
        asr = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        recall, missing = recall_and_missing(item["text"], asr)
        if recall < 0.7:
            with torch.no_grad():
                out2 = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=96)
            asr2 = processor.batch_decode(out2[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            r2, m2 = recall_and_missing(item["text"], asr2)
            if r2 > recall:
                asr, recall, missing = asr2, r2, m2
        gx, gsr = read_mono(item["gen"])
        lead, tail, excess = silence_profile(gx, gsr)
        dns = dnsmos_ovrl(gx, gsr, dnsmos_mod)
        rows.append({"idx": item["idx"], "text": item["text"], "seed": item["seed"], "gen": item["gen"],
                     "asr": asr, "recall": round(recall, 3), "missing": missing,
                     "lead": round(lead, 2), "tail": round(tail, 2), "excess": round(excess, 2),
                     "dnsmos": round(dns, 2)})
        print(f"c{rows[-1]['idx']:02d} s{item['seed']}: recall={recall:.2f} excess={excess:.2f} dns={dns:.2f} "
              f"missing={missing[:3]} | {asr[:60]}", flush=True)

    os.makedirs(os.path.join(args.dir, "best"), exist_ok=True)
    picked = []
    for idx in sorted({r["idx"] for r in rows}):
        cand = [r for r in rows if r["idx"] == idx]
        best = sorted(cand, key=lambda r: (-r["recall"], r["excess"], -r["dnsmos"]))[0]
        dst = os.path.join(args.dir, "best", f"best{idx:02d}.wav")
        shutil.copy(best["gen"], dst)
        best = dict(best, best_file=dst)
        picked.append(best)
        others = [f"s{c['seed']}:r{c['recall']}/e{c['excess']}" for c in cand if c is not best]
        print(f"PICK c{idx:02d}: seed={best['seed']} recall={best['recall']} excess={best['excess']} | others {others}",
              flush=True)

    json.dump(rows, open(os.path.join(args.dir, "scoring.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(picked, open(os.path.join(args.dir, "picked.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("BESTOFN_PICK_DONE", flush=True)


if __name__ == "__main__":
    main()
