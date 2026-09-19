"""Авто-гейт эмоций: Aniemore WavLM (82%, локально, CPU/GPU) на наших сэмплах.
Сравнивает ЦЕЛЕВУЮ эмоцию с предсказанной; отчёт JSON.

Использование:
  python emotion_gate.py --manifest <results.json c полями emotion+file> --out <out.json>
"""
import argparse
import json
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train\thirdparty\aniemore_pkg")

# маппинг наших лейблов -> классы Aniemore
LABEL_MAP = {
    "happy": "happiness", "sad": "sadness", "angry": "anger", "anger": "anger",
    "fear": "fear", "fearful": "fear", "disgust": "disgust", "disgusted": "disgust",
    "surprise": "enthusiasm", "surprised": "enthusiasm", "excited": "enthusiasm",
    "neutral": "neutral", "calm": "neutral",
    # родные классы Aniemore/RESD — identity
    "happiness": "happiness", "sadness": "sadness", "enthusiasm": "enthusiasm",
    # whisper/laughing/arrogance — отдельных классов нет: не оцениваем
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from aniemore.recognizers.voice import VoiceRecognizer
    from aniemore.models import HuggingFaceModel

    vr = VoiceRecognizer(model=HuggingFaceModel.Voice.WavLM, device=args.device)
    rows = json.load(open(args.manifest, encoding="utf-8"))

    out = []
    hits, total = 0, 0
    for r in rows:
        target = LABEL_MAP.get((r.get("emotion") or "").lower())
        if not target or not r.get("file"):
            out.append({**r, "emotion_gate": "n/a"})
            continue
        try:
            probs = vr.recognize(r["file"], return_single_label=False)
            probs = {k: float(v) for k, v in probs.items()}
            pred = max(probs, key=probs.get)
            ok = pred == target
            hits += ok
            total += 1
            out.append({**r, "emotion_gate": "hit" if ok else "miss",
                        "pred": pred, "target": target, "probs": probs})
        except Exception as e:
            out.append({**r, "emotion_gate": f"error {type(e).__name__}"})
        if (len(out) % 10) == 0:
            print(f"[{len(out)}/{len(rows)}] acc={hits}/{total}", flush=True)

    acc = round(hits / total, 4) if total else None
    json.dump({"accuracy": acc, "n": total, "rows": out},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"EMOTION_GATE_DONE acc={acc} n={total}")


if __name__ == "__main__":
    main()
