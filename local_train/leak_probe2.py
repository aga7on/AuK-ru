"""Leak probe 2: tighter durations + truncated refs (минимизация договорки хвоста рефа)."""
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
OUT = os.path.join(AUK, "local_tests", "leak_probe2")

T1 = "Съёмка фильма начнётся в конце месяца."
T2 = "Объёмный звук наполнил комнату."
T3 = "Всплеск эмоций был неожиданным."


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
    L = os.path.join(AUK, "local_tests")
    ref_ru = os.path.join(L, "ref_ru.wav")
    ref_m = os.path.join(L, "ref_male2.wav")
    ref_f = os.path.join(L, "ref_female2.wav")
    ref_ru_3 = truncate_ref(ref_ru, 3.0, os.path.join(OUT, "ref_ru_3s.wav"))
    ref_m_3 = truncate_ref(ref_m, 3.0, os.path.join(OUT, "ref_male2_3s.wav"))
    ref_f_4 = truncate_ref(ref_f, 4.0, os.path.join(OUT, "ref_female2_4s.wav"))

    runs = [
        ("ru_full_d3.4", ref_ru, T1, 3.4, 1234),
        ("ru_full_d3.0", ref_ru, T1, 3.0, 1234),
        ("ru_r3_d3.0", ref_ru_3, T1, 3.0, 1234),
        ("ru_r3_d2.6", ref_ru_3, T1, 2.6, 1234),
        ("m_full_d3.0", ref_m, T2, 3.0, 1234),
        ("m_r3_d2.6", ref_m_3, T2, 2.6, 1234),
        ("f_full_d3.2", ref_f, T3, 3.2, 1234),
        ("f_r4_d2.8", ref_f_4, T3, 2.8, 1234),
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
        out = os.path.join(OUT, f"leak2_{name}.wav")
        save_audio(audio, sr, out)
        heard = quality.transcribe(audio, sr)
        m = quality.wer_metrics(text, heard)
        rows.append({"name": name, "dur": secs, "ref_s": round(len(sf.read(ref)[0]) / sf.read(ref)[1], 2),
                     "wer": round(m["wer"], 2), "ins": m["ins"], "heard": heard})
        print(f"{name:<14} wer={m['wer']:.2f} ins={m['ins']} | {heard[:70]}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "leak_probe2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("LEAK_PROBE2_DONE")


if __name__ == "__main__":
    main()
