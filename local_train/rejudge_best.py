"""Re-judge stage-0 evaluation dirs with the accent-aware judge; pick the best checkpoint."""
import json
import os
import shutil
import sys

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

AUK = r"G:\AI\AuK"
LOCAL = os.path.join(AUK, "local_train")
sys.path.insert(0, LOCAL)
sys.path.insert(0, os.path.join(AUK, "src"))

from auto_controller import judge_means, score_results  # noqa: E402
from judge_eval import TEMPLATE, dnsmos_ovrl, f0_median, parse_json, read_mono  # noqa: E402

RUN = os.path.join(LOCAL, "run_ru")
EVAL_ROOT = os.path.join(RUN, "evals")
MERGED = os.path.join(RUN, "merged")
QWEN_PATH = os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B")
ORDER = ["base_u0", "lora_u750", "lora_u1250", "lora_u2750", "lora_u3000", "lora_u3500", "lora_u4000"]


def main():
    from speechmos import dnsmos as dnsmos_mod

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map="cuda:1",
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_PATH)

    table = {}
    for name in ORDER:
        d = os.path.join(EVAL_ROOT, name)
        man_path = os.path.join(d, "manifest.json")
        if not os.path.exists(man_path):
            print(f"{name}: no manifest, skip")
            continue
        man = json.load(open(man_path, encoding="utf-8"))
        results = []
        for item in man:
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio": item["ref"]},
                {"type": "audio", "audio": item["gen"]},
                {"type": "text", "text": TEMPLATE.format(text=item["text"])},
            ]}]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=prompt, audio=audios, images=images, videos=videos,
                               return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=192)
            reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            judge = parse_json(reply)
            gx, gsr = read_mono(item["gen"])
            rx, rsr = read_mono(item["ref"])
            obj = {
                "gen_dur": round(len(gx) / gsr, 2),
                "ref_dur": round(len(rx) / rsr, 2),
                "dur_ratio": round((len(gx) / gsr) / max(len(rx) / rsr, 0.01), 2),
                "gen_f0": round(f0_median(gx, gsr), 1),
                "ref_f0": round(f0_median(rx, rsr), 1),
                "gen_dnsmos": round(dnsmos_ovrl(gx, gsr, dnsmos_mod), 2),
            }
            results.append({"idx": item["idx"], "text": item["text"], "gen": item["gen"], "ref": item["ref"],
                            "judge": judge, "objective": obj, "raw_reply": reply[:500]})
        score, per = score_results(results)
        means = judge_means(results)
        table[name] = {"score": round(score, 4), "means": means, "results": results}
        print(f"{name:12s} score={score:.4f} overall={means['overall']:.1f} accent={means['accent']:.1f} "
              f"nat={means['naturalness']:.1f} sim={means['voice_similarity']:.1f} "
              f"text={means['text_matches']}/6 gender={means['same_gender']}/6", flush=True)

    lora = {k: v for k, v in table.items() if k.startswith("lora")}
    if not lora:
        print("no lora evals found")
        return
    best_name = max(lora, key=lambda k: lora[k]["score"])
    best_update = int(best_name.split("_u")[1])
    src = os.path.join(MERGED, f"auk_ru_{best_update}.safetensors")
    dst = os.path.join(RUN, "auk_ru_best.safetensors")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"best = {best_name} score={lora[best_name]['score']} -> {dst}")
    else:
        print(f"best = {best_name} but merged file {src} missing")

    summary = {
        "best": best_name,
        "update": best_update,
        "score": lora[best_name]["score"],
        "means": lora[best_name]["means"],
        "table": {k: {"score": v["score"], "means": v["means"]} for k, v in table.items()},
    }
    with open(os.path.join(RUN, "rejudge_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RUN, "BEST_v2.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({k: summary[k] for k in ("best", "update", "score", "means")},
                           ensure_ascii=False, indent=1))
    print("REJUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
