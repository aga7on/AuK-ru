"""Сборка s2-датасета v2 (по ревью). Small-пакет по умолчанию: 500 no-ref + 200 пар.

- единый split_manifest.jsonl; дедуп: md5 аудио + нормализованный текст ≤2/кластер;
- верификация цели: GigaAM wer<=0.12, ins==0, recall>=0.85, cer<=0.12;
- чистые ударения (без респеллинга); клонирование: явная «same voice» инструкция,
  проверка рефа (длительность/речь) и идентичности пары (embed sim >= 0.6);
- val: no-ref из dev + пары из dev-кластеров.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
OUT = os.path.join(LOCAL, "data_s2")
sys.path.insert(0, LOCAL)
sys.path.insert(0, r"G:\AI\AuK\src")

TEMPLATES_NOREF = [
    "Say the following in Russian with clear, natural pronunciation: '{t}'",
    "Speak the following Russian text aloud in a natural voice: '{t}'",
    "Pronounce the following in Russian: '{t}'",
    "Read the following Russian sentence out loud: '{t}'",
]
TEMPLATES_CLONE = [
    "Say the following with the same voice: '{t}'",
    "Reproduce the reference voice and say in Russian: '{t}'",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noref", type=int, default=500)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--val-noref", type=int, default=150)
    ap.add_argument("--val-pairs", type=int, default=50)
    ap.add_argument("--recall-min", type=float, default=0.85)
    ap.add_argument("--wer-max", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(LOCAL, "data_s2"))
    args = ap.parse_args()

    import numpy as np
    from ru_metrics import text_metrics, transcribe_path
    from auk.infer.infer_gradio import _accentize_ru
    from speaker_embed import embed_path

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    stats = defaultdict(int)
    t0 = time.time()

    stress_cache = {}

    def stress(text):
        key = text[:400]
        if key not in stress_cache:
            try:
                stress_cache[key] = _accentize_ru(text, use_lexicon=False)
            except Exception:
                stress_cache[key] = text
        return stress_cache[key]

    def content_md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(4_000_000))
        return h.hexdigest()

    def verify(path, text):
        heard, err = transcribe_path(path)
        if err or not heard.strip():
            return False, f"asr_fail:{err or 'empty'}"
        m = text_metrics(text, heard)
        if m["insertions"]:
            return False, "extra_words"
        if m["wer"] > args.wer_max or m["recall"] < args.recall_min or m["cer"] > 0.12:
            return False, f"mismatch(w={m['wer']:.2f},r={m['recall']:.2f})"
        return True, ""

    content_seen = set()
    text_seen = defaultdict(int)

    def add_item(items, text, target, ref=None, split="train", meta=None):
        ckey = hashlib.md5(text[:120].lower().encode("utf-8")).hexdigest()
        if text_seen[ckey] >= 2:
            stats["drop_text_cap"] += 1
            return False
        h = content_md5(target)
        if h in content_seen:
            stats["drop_content_dup"] += 1
            return False
        ok, why = verify(target, text)
        if not ok:
            stats[f"drop_{why}"] += 1
            return False
        t = stress(text)
        tmpl = rng.choice(TEMPLATES_NOREF if ref is None else TEMPLATES_CLONE)
        instr = tmpl.format(t=t)
        user_content = [{"type": "text", "text": instr}]
        if ref:
            user_content.append({"type": "audio", "audio": ref})
        try:
            import soundfile as sf
            info = sf.info(target)
            dur = round(info.frames / info.samplerate, 3)
        except Exception:
            dur = None
        row = {"duration": dur, "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "audio", "audio_url": target}]},
        ], "split": split, "meta": meta or {}}
        items.append(row)
        text_seen[ckey] += 1
        content_seen.add(h)
        stats[f"ok_{split}_{'pair' if ref else 'noref'}"] += 1
        return True

    # ---------- стриминг манифеста: подсчёт + кандидаты ----------
    prio_count = base_count = 0
    dev_items = []
    with open(os.path.join(CORPUS, "split_manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["split"] == "dev":
                dev_items.append(o)
                continue
            if any(t in ("numerals", "clusters", "soft_sign") for t in o.get("tags", [])):
                prio_count += 1
            else:
                base_count += 1
    print(f"train prio={prio_count} base={base_count} dev={len(dev_items)}", flush=True)

    def want(occ, total, need, rngx):
        if total <= need:
            return True
        return rngx.random() < need / total

    rx = random.Random(args.seed)
    noref_sample = []
    with open(os.path.join(CORPUS, "split_manifest.jsonl"), encoding="utf-8") as f:
        occ_p = occ_b = 0
        for line in f:
            o = json.loads(line)
            if o["split"] != "train":
                continue
            is_prio = any(t in ("numerals", "clusters", "soft_sign") for t in o.get("tags", []))
            if is_prio:
                if occ_p < int(args.noref * 0.4) and rx.random() < 1.0:
                    noref_sample.append(o)
                    occ_p += 1
                else:
                    occ_p += 1
            else:
                if len([x for x in noref_sample if not any(t in ("numerals", "clusters", "soft_sign") for t in x.get("tags", []))]) < args.noref - int(args.noref * 0.4) and rx.random() < 0.02:
                    noref_sample.append(o)
                occ_b += 1
    print(f"noref candidates: {len(noref_sample)}", flush=True)

    train_rows = []
    val_rows = []
    for k, r in enumerate(noref_sample):
        if len(train_rows) >= args.noref:
            break
        if not os.path.exists(r["p"]):
            stats["drop_missing"] += 1
            continue
        add_item(train_rows, r["t"], r["p"], split="train")
        if (k + 1) % 200 == 0:
            print(f"  noref {k+1}/{len(noref_sample)} train_rows={len(train_rows)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    pairs_rows = []
    with open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8") as f:
        for line in f:
            pairs_rows.append(json.loads(line))
    rng.shuffle(pairs_rows)
    made = 0
    for pr in pairs_rows:
        if made >= args.pairs:
            break
        if not (os.path.exists(pr["target"]) and os.path.exists(pr["ref"])):
            stats["drop_missing"] += 1
            continue
        try:
            er, et = embed_path(pr["ref"]), embed_path(pr["target"])
            sim = float(np.dot(er, et))
        except Exception:
            sim = -1.0
        if sim < 0.6:
            stats["drop_identity"] += 1
            continue
        ref_d = float(pr.get("ref_d", 0))
        if not (2.0 <= ref_d <= 10.0):
            stats["drop_ref_duration"] += 1
            continue
        ok, why = verify(pr["target"], pr["target_t"])
        if not ok:
            stats[f"drop_pair_{why}"] += 1
            continue
        add_item(train_rows, pr["target_t"], pr["target"], ref=pr["ref"], split="train",
                 meta={"pair_sim": round(sim, 3), "ref_d": ref_d})
        made += 1
        if made % 50 == 0:
            print(f"  pairs {made}/{args.pairs} ({(time.time()-t0)/60:.1f}m)", flush=True)

    rng.shuffle(train_rows)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # val no-ref (dev) — несколько попыток на кластер до цели
    dev_by_cluster = defaultdict(list)
    for r in dev_items:
        dev_by_cluster[r["cluster"]].append(r)
    val_rows = []
    v_made = 0
    keys = list(dev_by_cluster)
    rng.shuffle(keys)
    attempts = defaultdict(int)
    for cl in keys:
        pool = dev_by_cluster[cl]
        rng.shuffle(pool)
        for r in pool[:3]:
            if v_made >= args.val_noref:
                break
            attempts[cl] += 1
            if not os.path.exists(r["p"]):
                continue
            if add_item(val_rows, r["t"], r["p"], split="dev"):
                v_made += 1

    # val pairs (dev-кластеры ≥8, разные тексты, до цели)
    v_pairs = 0
    for cl in keys:
        if v_pairs >= args.val_pairs:
            break
        pool = dev_by_cluster[cl]
        if len(pool) < 4:
            continue
        rng.shuffle(pool)
        tried = 0
        for _ in range(3):
            if v_pairs >= args.val_pairs or tried >= 3 or len(pool) < 2:
                break
            a, b = rng.sample(pool, 2)
            tried += 1
            if a["t"][:40].lower() == b["t"][:40].lower():
                continue
            try:
                ea, eb = embed_path(a["p"]), embed_path(b["p"])
                sim = float(np.dot(ea, eb))
            except Exception:
                sim = -1.0
            if sim < 0.6:
                stats["drop_identity_val"] += 1
                continue
            if not os.path.exists(a["p"]) or not os.path.exists(b["p"]):
                continue
            ok, why = verify(b["p"], b["t"])
            if not ok:
                stats[f"drop_valpair_{why}"] += 1
                continue
            add_item(val_rows, b["t"], b["p"], ref=a["p"], split="dev",
                     meta={"pair_sim": round(sim, 3)})
            v_pairs += 1

    with open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats.update({"train_rows": len(train_rows), "val_rows": len(val_rows),
                  "val_pairs": v_pairs, "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(dict(stats), open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print("S2_BUILD_V2_DONE")


if __name__ == "__main__":
    main()
