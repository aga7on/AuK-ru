# -*- coding: utf-8 -*-
"""S13: human pairwise A/B benchmark — построение слепого набора.

Дизайн (ROADMAP S13): только A/B-выбор, никаких «оценок 1–10».
Пары: (s7@5750 vs s8@7750) на одинаковых текстах/референсах; порядок A/B рандомизирован,
маппинг хранится в SECRET_pair_map.csv (НЕ ОТКРЫВАТЬ до конца прослушиваний, не коммитить).

Набор v1 (50 пар):
  - 20 TTS-текстов из hard_cases_ru.jsonl (по 2 на категорию: number/money/date/phone/units);
  - 15 clone-текстов из clone100-промптов;
  - 15 эмо-текстов (5 эмоций × 3 фразы, как эмо-протокол).

Выход: local_tests/pairwise_v1/{a.wav,b.wav,LISTEN.csv} + SECRET_pair_map.csv.
Генерация требует оба merged-чекпойнта; скрипт идемпотентен (пропускает готовые wav).

usage: python s13_pairwise_build.py --gen   (генерация; без --gen только список пар)
"""
import argparse
import csv
import json
import os
import random
import sys

AUK = r"G:\AI\AuK"
OUT = os.path.join(AUK, "local_tests", "pairwise_v1")
SECRET = os.path.join(OUT, "SECRET_pair_map.csv")
SEED = 13

CKPTS = {
    "s7": (os.path.join(AUK, "local_train", "run_s7", "merged", "auk_s7_5750.safetensors"),
           os.path.join(AUK, "local_train", "run_s7", "merged", "config.yaml")),
    "s8": (os.path.join(AUK, "local_train", "run_s8", "merged", "auk_s8_7750.safetensors"),
           os.path.join(AUK, "local_train", "run_s8", "merged", "config.yaml")),
}


def build_pairs():
    rng = random.Random(SEED)
    pairs = []
    # 1) hard-case TTS (frontend-категории)
    bench = [json.loads(l) for l in open(os.path.join(AUK, "local_train", "hard_cases", "hard_cases_ru.jsonl"), encoding="utf-8") if l.strip()]
    by_cat = {}
    for r in bench:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in ("number_case", "money", "date_time", "phone", "units_frac"):
        for r in by_cat.get(cat, [])[:2]:
            pairs.append({"kind": "tts_hard", "cat": cat, "text": r["text"]})
    # 2) clone (из clone100 prompts — первые 15)
    c100 = json.load(open(os.path.join(AUK, "local_tests", "s7_5750_clone100", "results.json"), encoding="utf-8"))
    for r in [x for x in c100 if x.get("status") == "ok"][:15]:
        pairs.append({"kind": "clone", "text": r["text"], "ref": r["ref"]})
    # 3) emotion (5 эмоций × 3 фразы — как эмо-протокол)
    emo = json.load(open(os.path.join(AUK, "local_tests", "emotion_ru_s7_5750", "results.json"), encoding="utf-8"))
    seen = set()
    for r in emo:
        key = (r["emotion"], r["text"])
        if key in seen or r["seed"] != 7:
            continue
        seen.add(key)
        pairs.append({"kind": "emotion", "emotion": r["emotion"], "text": r["text"], "ref": r["ref"]})
        if len(seen) >= 15:
            break
    # рандомизируем порядок A/B
    for i, p in enumerate(pairs):
        p["id"] = f"pr{i:02d}"
        p["a_variant"] = rng.choice(("s7", "s8"))
        p["b_variant"] = "s8" if p["a_variant"] == "s7" else "s7"
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pairs = build_pairs()
    print("pairs:", len(pairs), "kinds:", {k: sum(1 for p in pairs if p['kind']==k) for k in set(p['kind'] for p in pairs)})

    # SECRET map
    with open(SECRET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "a_variant", "b_variant"])
        for p in pairs:
            w.writerow([p["id"], p["a_variant"], p["b_variant"]])

    # LISTEN.csv (публичная часть: id, kind, текст, пути a/b)
    with open(os.path.join(OUT, "LISTEN.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "text", "a_wav", "b_wav", "choice", "more_natural", "better_diction", "closer_to_ref", "better_emotion", "would_use"])
        for p in pairs:
            w.writerow([p["id"], p["kind"], p["text"], f"{p['id']}_a.wav", f"{p['id']}_b.wav", "", "", "", "", "", ""])
    print("LISTEN.csv + SECRET_pair_map.csv written (SECRET не открывать/не коммитить)")

    if not args.gen:
        print("dry-run: список пар готов; --gen для генерации (нужны оба merged)")
        return

    import soundfile as sf
    sys.path.insert(0, os.path.join(AUK, "src"))
    sys.path.insert(0, os.path.join(AUK, "local_train"))
    from auk.infer.infer_auk import AukInfer, save_audio
    from auk.infer.quality import normalize_rms, limit_peak
    from auk.infer.ru_frontend import to_speakable

    engines = {}
    for name, (ck, cfg) in CKPTS.items():
        if not os.path.exists(ck):
            print(f"SKIP gen: missing {ck}")
            return
        engines[name] = AukInfer(config_path=cfg, ckpt_path=ck,
                                 qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                                 cpu_offload=True, device=args.device, dtype="bf16")

    pack = json.load(open(os.path.join(AUK, "local_tests", "eval_pack", "pack.json"), encoding="utf-8"))
    tts_ref = next(e["ref"] for e in pack if e.get("kind") == "clone" and e["id"] == "clone08")

    def gen_one(eng, p, side):
        out = os.path.join(OUT, f"{p['id']}_{side}.wav")
        if os.path.exists(out):
            return out
        if p["kind"] == "tts_hard":
            instr = "Say the following in Russian with clear, natural pronunciation: '%s'" % to_speakable(p["text"])
            ref = tts_ref
        elif p["kind"] == "clone":
            instr = "Reproduce the reference voice and say in Russian: '%s'" % p["text"].replace("+", "")
            ref = p["ref"]
        else:
            instr = "Reproduce the reference voice and say in Russian with a %s tone: '%s'" % (p["emotion"], p["text"])
            ref = p["ref"]
        content = [{"type": "text", "text": instr}]
        audio_kw = {}
        if p["kind"] != "tts_hard":
            content.append({"type": "audio", "audio": ref})
            audio_kw["audio"] = ref
        audio, sr = eng.generate([{"role": "user", "content": content}],
                                 gen_seconds=8.0 if p["kind"] == "tts_hard" else float(sf.info(ref).duration) + 0.5,
                                 nfe=64, cfg_strength=2.0, seed=13, **audio_kw)
        audio = limit_peak(normalize_rms(audio))
        save_audio(audio, sr, out)
        return out

    n = 0
    for p in pairs:
        for side, var in (("a", p["a_variant"]), ("b", p["b_variant"])):
            gen_one(engines[var], p, side)
            n += 1
        print(f"[{n}/{len(pairs)*2}] {p['id']} done", flush=True)
    print("PAIRWISE_GEN_DONE", len(pairs))


if __name__ == "__main__":
    main()
