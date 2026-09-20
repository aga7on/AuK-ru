# -*- coding: utf-8 -*-
"""S15 (вариант 3): маршрутизация адаптеров по типу инструкции — решение emotion/neutral trade-off
БЕЗ обучения и без композиции весов (S15_RESULTS: композиция α1.0+α0.5 провалила гейт 0/5).

Принцип (обе ветви измерены полными протоколами):
  emotion-инструкция → s7@5750 (эмо годен 48.3%, TTS WER 0.077, first_ok 0.812)
  neutral/clone/tts   → s5@4500 (judge clone 8.20, phonetics 8.06, clone WER 0.055)

Пакетная реализация: запросы группируются, каждая группа генерируется одним движком
(2 загрузки модели на пакет, а не на запрос). Для real-time hot-swap нужен resident-base
загрузчик (S15_DESIGN, этап 1) — отмечается как следующая работа.

usage:
  python route_infer.py --requests requests.jsonl --out local_tests/routed
  # строка requests.jsonl: {"id", "text", "ref"?, "emotion"?}
  # либо --demo (8 встроенных запросов: 4 neutral + 4 emotion)
"""
import argparse
import json
import os
import sys

import soundfile as sf

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
ADAPTERS = {
    "emotion": (os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
                os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml")),
    "neutral": (os.path.join(AUK, "local_train", "run_s5", "merged", "auk_s5_4500.safetensors"),
                os.path.join(AUK, "local_train", "run_s5", "merged", "config.yaml")),
}
EMOTIONS = {"happy", "sad", "angry", "fearful", "excited", "whispering", "laughing",
            "surprised", "disgusted"}

DEMO = [
    {"id": "n0", "text": "Завтра состоится встреча по проекту в десять утра."},
    {"id": "n1", "text": "Пожалуйста, не опаздывайте на встречу."},
    {"id": "n2", "text": "Мы проверили все документы и отправили отчёт."},
    {"id": "n3", "text": "Сельдь под шубой — традиционное новогоднее блюдо.", "ref": "clone01"},
    {"id": "e0", "text": "Ура, мы наконец-то закончили этот проект!", "emotion": "happy", "ref": "clone01"},
    {"id": "e1", "text": "Мне очень жаль, но всё потеряно.", "emotion": "sad", "ref": "clone08"},
    {"id": "e2", "text": "Как ты мог так поступить со мной?", "emotion": "angry", "ref": "clone08"},
    {"id": "e3", "text": "Там за дверью кто-то есть, я слышу шаги.", "emotion": "fearful", "ref": "clone01"},
]


def route(req):
    """emotion → s7 (48.3% годен); neutral+ref (clone/phonetics) → s5 (judge 8.20/8.06);
    neutral без ref (простой TTS) → s7 (WER 0.077/first_ok 0.812 лучше s5 0.081/0.750)."""
    e = (req.get("emotion") or "").lower()
    if e in EMOTIONS:
        return "emotion"
    txt = (req.get("text") or "").lower()
    for k in (" tone:", "with a happy", "with a sad", "шёпотом", "весело", "грустно", "злость"):
        if k in txt:
            return "emotion"
    if req.get("ref"):
        return "neutral"
    return "emotion"  # нейтральный TTS без рефа: профиль s7 лучше по TTS-метрикам


def build_instruction(req, speakable):
    emo = (req.get("emotion") or "").lower()
    if emo in EMOTIONS:
        if req.get("ref"):
            return (f"Reproduce the reference voice and say in Russian with a "
                    f"{emo} tone: '{speakable}'")
        return f"Say the following in Russian with a {emo} tone: '{speakable}'"
    if req.get("ref"):
        return f"Reproduce the reference voice and say in Russian: '{speakable}'"
    return f"Say the following in Russian with clear, natural pronunciation: '{speakable}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=os.path.join(AUK, "local_tests", "routed"))
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        reqs = [dict(r) for r in DEMO]
    else:
        reqs = [json.loads(l) for l in open(args.requests, encoding="utf-8") if l.strip()]

    # resolve refs (demo ids clone01/clone08 → eval_pack)
    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    ref_by_id = {e["id"]: e["ref"] for e in pack if e.get("kind") == "clone"}
    for r in reqs:
        if r.get("ref") in ref_by_id:
            r["ref"] = ref_by_id[r["ref"]]

    groups = {"emotion": [], "neutral": []}
    for r in reqs:
        groups[route(r)].append(r)
    print("routing:", {k: len(v) for k, v in groups.items()})

    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from auk.infer.ru_frontend import to_speakable

    results = []
    for kind in ("neutral", "emotion"):
        rs = groups[kind]
        if not rs:
            continue
        ck, cfg = ADAPTERS[kind]
        print(f"loading {kind} adapter: {os.path.basename(ck)}")
        eng = AukInfer(config_path=cfg, ckpt_path=ck,
                       qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                       cpu_offload=True, device=args.device, dtype="bf16")
        for r in rs:
            speak = to_speakable(r["text"])
            instr = build_instruction(r, speak)
            content = [{"type": "text", "text": instr}]
            kw = {}
            ref = r.get("ref")
            if ref:
                content.append({"type": "audio", "audio": ref})
                kw["audio"] = ref
                gs = float(sf.info(ref).duration) + 0.5
            else:
                gs = max(3.5, min(14.0, 1.0 + 0.09 * len(speak)))
            out = os.path.join(args.out, f"{r['id']}_{kind}.wav")
            try:
                audio, sr = eng.generate([{"role": "user", "content": content}],
                                         gen_seconds=gs, nfe=64, cfg_strength=2.0,
                                         seed=r.get("seed", 7), **kw)
                audio = limit_peak(normalize_rms(audio))
                save_audio(audio, sr, out)
                results.append({"id": r["id"], "adapter": kind, "emotion": r.get("emotion"),
                                "text": r["text"], "file": out, "status": "ok"})
            except Exception as e:
                results.append({"id": r["id"], "adapter": kind, "text": r["text"],
                                "status": f"error {type(e).__name__}"})
            print(f"  [{kind}] {r['id']} {results[-1]['status']}", flush=True)
        del eng

    json.dump(results, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["status"] == "ok")
    by = {}
    for r in results:
        by.setdefault(r["adapter"], [0, 0])
        by[r["adapter"]][0] += 1 if r["status"] == "ok" else 0
        by[r["adapter"]][1] += 1
    print(f"ROUTE_DONE ok={ok}/{len(results)} per_adapter={by}")


if __name__ == "__main__":
    main()
