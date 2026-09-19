"""Mitigation probe: ref-content leak on short targets.
Variants: longer target window, truncated reference, different seed.
"""
import json
import os
import sys

import soundfile as sf

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

from auk.infer.infer_auk import AukInfer, save_audio  # noqa: E402
from auk.infer.infer_gradio import _accentize_ru  # noqa: E402
from auk.infer import quality  # noqa: E402

MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
OUT = os.path.join(AUK, "local_tests", "leak_probe")

CASE1_TEXT = "Съёмка фильма начнётся в конце месяца."
CASE2_TEXT = "Объёмный звук наполнил комнату."


def est(text):
    return max(3.5, min(12.0, 1.0 + 0.075 * len(text)))


def truncate_ref(src, seconds, dst):
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sf.write(dst, x[: int(seconds * sr)], sr)
    return dst


def main():
    os.makedirs(OUT, exist_ok=True)
    engine = AukInfer(
        config_path=os.path.join(os.path.dirname(MERGED), "config.yaml"),
        ckpt_path=MERGED,
        qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
        cpu_offload=True,
        device="cuda:0",
        dtype="bf16",
    )

    ref_ru = os.path.join(AUK, "local_tests", "ref_ru.wav")
    ref_m = os.path.join(AUK, "local_tests", "ref_male2.wav")
    ref_ru_5 = truncate_ref(ref_ru, 5.0, os.path.join(OUT, "ref_ru_5s.wav"))
    ref_ru_3 = truncate_ref(ref_ru, 3.0, os.path.join(OUT, "ref_ru_3s.wav"))
    ref_m_4 = truncate_ref(ref_m, 4.0, os.path.join(OUT, "ref_male2_4s.wav"))

    runs = [
        ("ru_base_3.9_s1234", ref_ru, CASE1_TEXT, 3.9, 1234),
        ("ru_long_4.9_s1234", ref_ru, CASE1_TEXT, 4.9, 1234),
        ("ru_long_5.9_s1234", ref_ru, CASE1_TEXT, 5.9, 1234),
        ("ru_base_3.9_s7", ref_ru, CASE1_TEXT, 3.9, 7),
        ("ru_ref5s_3.9_s1234", ref_ru_5, CASE1_TEXT, 3.9, 1234),
        ("ru_ref3s_3.9_s1234", ref_ru_3, CASE1_TEXT, 3.9, 1234),
        ("m_base_3.5_s1234", ref_m, CASE2_TEXT, 3.5, 1234),
        ("m_long_4.5_s1234", ref_m, CASE2_TEXT, 4.5, 1234),
        ("m_long_5.5_s1234", ref_m, CASE2_TEXT, 5.5, 1234),
        ("m_ref4s_3.5_s1234", ref_m_4, CASE2_TEXT, 3.5, 1234),
    ]
    rows = []
    for name, ref, text, secs, seed in runs:
        body = _accentize_ru(text)
        instr = f"Say the following in Russian with clear, natural pronunciation: '{body}'"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=seed)
        out = os.path.join(OUT, f"leak_{name}.wav")
        save_audio(audio, sr, out)
        heard = quality.transcribe(audio, sr)
        m = quality.wer_metrics(text, heard)
        rows.append({"name": name, "dur": secs, "seed": seed, "wer": round(m["wer"], 2),
                     "ins": m["ins"], "heard": heard})
        print(f"{name:<22} wer={m['wer']:.2f} ins={m['ins']} | {heard[:75]}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "leak_probe.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("LEAK_PROBE_DONE")


if __name__ == "__main__":
    main()
