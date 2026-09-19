"""100-клоновый стресс-тест: любые случайные голоса из корпуса (вне train-пула),
генерация клона на русском + объективные замеры (WeSpeaker sim, GigaAM WER) + манифест для Гемини.

Этап 1 (этот скрипт): выбор 100 рефов (не в v2_after_identity train), генерация, замеры.
Этап 2: судья v3 по манифесту (внешний вызов).
"""
import argparse
import json
import os
import random
import sys
import time

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

CORPUS = r"G:\AI\kyutai-ru\data\ru_wav_mfa"
TRAIN_POOL = os.path.join(AUK, "local_train", "data_s2_full", "v2_after_identity", "train.jsonl")
OUT_DIR = os.path.join(AUK, "local_tests", "s5_clone100")
SEED = 1234

TEXTS = [
    "сколько стоит тысяча фунтов стерлингов в сбербанке?",
    "погода сегодня замечательная, пойдем гулять в парк",
    "он работает инженером на заводе уже пять лет",
    "мне кажется, этот фильм лучше предыдущего",
    "завтра утром я поеду в командировку в новосибирск",
    "на улице холодно, надо надеть теплую куртку",
    "она учится на третьем курсе медицинского университета",
    "мы купили новый холодильник и стиральную машину",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(SEED)

    # train-пул путей (рефы), чтобы исключить
    pool = set()
    for l in open(TRAIN_POOL, encoding="utf-8"):
        r = json.loads(l)
        for m in r["messages"]:
            for c in m["content"]:
                if c.get("type") == "audio":
                    a = c.get("audio") or c.get("audio_url")
                    if a:
                        pool.add(os.path.normcase(a))

    corpus = [os.path.join(CORPUS, f) for f in os.listdir(CORPUS) if f.endswith(".wav")]
    # только достаточно длинные и не в пуле
    import soundfile as sf
    candidates = []
    rng.shuffle(corpus)
    for f in corpus:
        if os.path.normcase(f) in pool:
            continue
        try:
            d = sf.info(f).duration
        except Exception:
            continue
        if 2.5 <= d <= 8.0:
            candidates.append(f)
        if len(candidates) >= args.n:
            break
    print(f"candidates={len(candidates)} (pool excluded: {len(pool)} refs)", flush=True)

    from auk.infer.infer_auk import AukInfer, save_audio
    engine = AukInfer(config_path=os.path.join(os.path.dirname(args.ckpt), "config.yaml"),
                      ckpt_path=args.ckpt,
                      qwen_path=os.path.join(AUK, "ckpts", "Qwen2.5-Omni-3B"),
                      cpu_offload=True, device=args.device, dtype="bf16")

    results = []
    t0 = time.time()
    for i, ref in enumerate(candidates):
        text = TEXTS[i % len(TEXTS)]
        instr = f"Reproduce the reference voice and say in Russian: '{text}'"
        out = os.path.join(args.out_dir, f"clone_{i:03d}.wav")
        try:
            content = [{"type": "text", "text": instr}, {"type": "audio", "audio": ref}]
            audio, sr = engine.generate([{"role": "user", "content": content}], audio=ref,
                                        gen_seconds=float(sf.info(ref).duration) + 0.5,
                                        nfe=64, cfg_strength=2.0, seed=SEED)
            save_audio(audio, sr, out)
            results.append({"id": f"clone_{i:03d}", "instruction": instr, "text": text,
                            "ref": ref, "file": out, "status": "ok"})
        except Exception as ex:
            results.append({"id": f"clone_{i:03d}", "instruction": instr, "text": text,
                            "ref": ref, "file": None, "status": f"error {type(ex).__name__}"})
        print(f"[{i+1}/{len(candidates)}] {results[-1]['status']} ({(time.time()-t0)/60:.1f}m)", flush=True)

    json.dump(results, open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"CLONE100_DONE ok={ok}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
