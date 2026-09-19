"""Leak probe 3: tighter windows for male2/female2 + instruction template influence."""
import json
import os
import sys

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

from auk.infer.infer_auk import AukInfer, save_audio  # noqa: E402
from auk.infer.infer_gradio import _accentize_ru  # noqa: E402
from auk.infer import quality  # noqa: E402

MERGED = os.path.join(AUK, "local_train", "run_ru_s1", "merged", "auk_ru_18000.safetensors")
OUT = os.path.join(AUK, "local_tests", "leak_probe3")

T2 = "Объёмный звук наполнил комнату."
T3 = "Всплеск эмоций был неожиданным."

TEMPLATES = [
    "Say the following in Russian with clear, natural pronunciation: '{b}'",
    "Speak the following Russian text aloud in a natural voice: '{b}'",
    "Pronounce the following in Russian: '{b}'",
    "Read the following Russian sentence out loud: '{b}'",
]


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
    ref_m = os.path.join(L, "ref_male2.wav")
    ref_f = os.path.join(L, "ref_female2.wav")

    runs = [
        ("m_d2.4", ref_m, T2, 2.4, 0),
        ("m_d2.2", ref_m, T2, 2.2, 0),
        ("f_d2.4", ref_f, T3, 2.4, 0),
        ("m_d3.0_tpl2", ref_m, T2, 3.0, 1),
        ("m_d3.0_tpl3", ref_m, T2, 3.0, 2),
        ("m_d3.0_tpl4", ref_m, T2, 3.0, 3),
    ]
    rows = []
    for name, ref, text, secs, tpl_idx in runs:
        body = _accentize_ru(text)
        instr = TEMPLATES[tpl_idx].format(b=body)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "audio": ref},
        ]}]
        audio, sr = engine.generate(messages, audio=ref, gen_seconds=secs, nfe=64,
                                    cfg_strength=2.0, seed=1234)
        out = os.path.join(OUT, f"leak3_{name}.wav")
        save_audio(audio, sr, out)
        heard = quality.transcribe(audio, sr)
        m = quality.wer_metrics(text, heard)
        rows.append({"name": name, "dur": secs, "tpl": tpl_idx, "wer": round(m["wer"], 2),
                     "ins": m["ins"], "heard": heard})
        print(f"{name:<14} wer={m['wer']:.2f} ins={m['ins']} | {heard[:70]}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "leak_probe3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("LEAK_PROBE3_DONE")


if __name__ == "__main__":
    main()
