"""Диагностика точного u0 провалившегося тех-прогона.

init tech-прогона = run_ru\auk_ru_best.safetensors (НЕ u10000!).
Проверяем: тот же пример («Сбер»), те же инструкции/настройки + 5 известных корректных примеров.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

INIT = os.path.join(AUK, "local_train", "run_ru", "auk_ru_best.safetensors")
OUT = os.path.join(AUK, "local_train", "diagnostics")


def main():
    from auk.infer.infer_auk import AukInfer, save_audio
    from gigaam_asr import transcribe

    engine = AukInfer(
        config_path=os.path.join(AUK, "local_train", "run_ru", "config.yaml"),
        ckpt_path=INIT,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:0",
        dtype="bf16",
    )

    # кейс 1: «Сбер» из провалившегося прогона
    row = json.load(open(os.path.join(OUT, "problem_row24.json"), encoding="utf-8"))
    content = row["row"]["messages"][0]["content"]
    raw_instr = content[0]["text"]
    ref = None
    for c in content:
        if c.get("type") == "audio":
            ref = c.get("audio")
    text = raw_instr.split(": '", 1)[-1].rstrip("'").rstrip("'")
    gen_secs = float(row["row"]["duration"] or 4.6)

    print(f"=== INIT CHECK: {os.path.basename(INIT)} ===")
    print(f"ref: {ref}")
    print(f"text: {text[:80]}")
    print(f"secs: {gen_secs}")

    results = []

    def run_case(tag, instruction, r, secs, seed=1234):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": r},
        ]}]
        audio, sr = engine.generate(messages, audio=r, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=seed)
        path = os.path.join(OUT, f"initchk_{tag}.wav")
        save_audio(audio, sr, path)
        heard = transcribe(path)
        results.append({"tag": tag, "heard": heard, "path": path})
        print(f"  {tag}: {heard[:80]}", flush=True)

    # тот же кейс, что был в провалившемся прогоне (шаблон «Reproduce the reference voice»)
    run_case("sber_reproduce", raw_instr, ref, gen_secs)
    # тот же кейс с продуктовым шаблоном
    run_case("sber_product", f"Say the following with the same voice: '{text}'", ref, gen_secs)

    # 5 известных корректных примеров (из data_s2_full, не пересекаются с этим кейсом)
    checked = 0
    with open(os.path.join(AUK, "local_train", "data_s2_full", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            if checked >= 5:
                break
            o = json.loads(line)
            tgt = o["messages"][1]["content"][0]["audio_url"]
            instr = o["messages"][0]["content"][0]["text"]
            c = o["messages"][0]["content"]
            r = c[1]["audio"] if len(c) > 1 else None
            if r is None:
                continue
            if not (os.path.exists(tgt) and os.path.exists(r)):
                continue
            t = instr.split(": '", 1)[-1].rstrip("'").rstrip("'")
            run_case(f"ref{checked+1}", f"Say the following with the same voice: '{t}'", r,
                     float(o.get("duration") or 4.0))
            checked += 1

    json.dump(results, open(os.path.join(OUT, "init_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("INIT_CHECK_DONE")


if __name__ == "__main__":
    main()
