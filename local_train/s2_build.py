"""Сборка пилотного s2-датасета (шаг 8):
- 15,000 no-ref TTS + 8,000 пар клонирования (реф≠цель из s2_pairs);
- верификация GigaAM (recall>=0.85), дедуп по эмбеддингам (cos>0.97), 4 обученных шаблона;
- val: 300+100 из dev-сплита (тот же сид сплита).
Выход: local_train/data_s2/{train,val}.jsonl + build_report.json
"""
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

TEMPLATES = [
    "Say the following in Russian with clear, natural pronunciation: '{t}'",
    "Speak the following Russian text aloud in a natural voice: '{t}'",
    "Pronounce the following in Russian: '{t}'",
    "Read the following Russian sentence out loud: '{t}'",
]

N_NOREF = 15000
N_PAIRS = 8000
N_VAL_NOREF = 300
N_VAL_PAIRS = 100
SEED = 7


def load_lines(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stress_cache():
    from auk.infer.infer_gradio import _accentize_ru

    cache = {}

    def get(t):
        key = t[:400]
        if key not in cache:
            try:
                cache[key] = _accentize_ru(t)
            except Exception:
                cache[key] = t
        return cache[key]

    return get


def main():
    import numpy as np
    sys.path.insert(0, LOCAL)
    from ru_metrics import text_metrics, transcribe_path

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    sel = load_lines(os.path.join(CORPUS, "s2_selection.jsonl"))
    pairs = load_lines(os.path.join(CORPUS, "s2_pairs.jsonl"))
    labels = np.load(os.path.join(CORPUS, "speakers.npy"))
    print(f"selection={len(sel)} pairs={len(pairs)}", flush=True)

    # dev-сплит (те же 8%/4% по кластерам, что в corpus_select)
    by_spk = defaultdict(list)
    idx = load_lines(os.path.join(CORPUS, "corpus_index.jsonl"))
    for i, o in enumerate(idx):
        spk = int(labels[i]) if i < len(labels) else -1
        if spk >= 0 and 2.0 <= float(o.get("d", 0)) <= 8.0:
            by_spk[spk].append(i)
    spk_ids = sorted(by_spk)
    rng2 = random.Random(SEED)
    rng2.shuffle(spk_ids)
    n_dev = max(1, int(len(spk_ids) * 0.08))
    dev_spk = set(spk_ids[:n_dev])
    dev_idx = {i for s in dev_spk for i in by_spk[s]}

    # выборка no-ref: приоритет редким тегам
    prio = [r for r in sel if any(t in ("numerals", "clusters", "soft_sign") for t in r.get("tags", []))]
    base = [r for r in sel if r not in prio]
    rng.shuffle(prio)
    rng.shuffle(base)
    n_prio = min(len(prio), int(N_NOREF * 0.4))
    noref = prio[:n_prio] + base[: N_NOREF - n_prio]
    print(f"noref pool: prio={n_prio} base={len(noref)-n_prio}", flush=True)

    pairs_sel = pairs[:]
    rng.shuffle(pairs_sel)
    pairs_sel = pairs_sel[:N_PAIRS]

    # дедуп по эмбеддингам (cos>0.97) внутри выборки
    def emb_for(i):
        sid, off = divmod(i, 20000)
        z = np.load(os.path.join(CORPUS, "emb", f"emb_{sid:05d}.npz"))
        if off >= len(z["ok"]) or not z["ok"][off]:
            return None
        return z["vecs"][off]

    def dedup(items, key="i", get_emb=True):
        seen = []
        out = []
        vecs = {}
        for it in items:
            i = it[key] if key in it else None
            v = emb_for(i) if (get_emb and i is not None) else None
            if v is not None:
                dup = False
                for w in seen:
                    if float(np.dot(v, w)) > 0.97:
                        dup = True
                        break
                if dup:
                    continue
                seen.append(v)
            out.append(it)
        return out

    noref = dedup(noref)
    print(f"after dedup noref: {len(noref)}", flush=True)

    get_stress = stress_cache()

    def verify(path, text):
        heard, err = transcribe_path(path)
        if err:
            return False, 0.0
        m = text_metrics(text, heard)
        return m["recall"] >= 0.85, m["recall"]

    def make_row(text, audio_path, ref_path=None):
        t = get_stress(text)
        tmpl = rng.choice(TEMPLATES)
        instr = tmpl.format(t=t)
        user_content = [{"type": "text", "text": instr}]
        if ref_path:
            user_content.append({"type": "audio", "audio": ref_path})
        return {"duration": None, "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "audio", "audio_url": audio_path}]},
        ]}

    rows = []
    stats = {"noref_ok": 0, "noref_drop": 0, "pair_ok": 0, "pair_drop": 0}
    t0 = time.time()
    for k, r in enumerate(noref):
        p = r["p"]
        if not os.path.exists(p):
            stats["noref_drop"] += 1
            continue
        ok, rec = verify(p, r["t"])
        if not ok:
            stats["noref_drop"] += 1
            continue
        row = make_row(r["t"], p)
        row["duration"] = float(r["d"])
        rows.append(row)
        stats["noref_ok"] += 1
        if (k + 1) % 500 == 0:
            print(f"  noref {k+1}/{len(noref)} ok={stats['noref_ok']} ({(time.time()-t0)/60:.1f} min)", flush=True)

    for k, pr in enumerate(pairs_sel):
        if not (os.path.exists(pr["target"]) and os.path.exists(pr["ref"])):
            stats["pair_drop"] += 1
            continue
        ok, rec = verify(pr["target"], pr["target_t"])
        if not ok:
            stats["pair_drop"] += 1
            continue
        row = make_row(pr["target_t"], pr["target"], ref_path=pr["ref"])
        row["duration"] = float(pr["target_d"])
        rows.append(row)
        stats["pair_ok"] += 1
        if (k + 1) % 500 == 0:
            print(f"  pairs {k+1}/{len(pairs_sel)} ok={stats['pair_ok']} ({(time.time()-t0)/60:.1f} min)", flush=True)

    rng.shuffle(rows)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # val: из dev-сплита
    val_rows = []
    dev_pool = [i for i in dev_idx]
    rng.shuffle(dev_pool)
    made = 0
    for i in dev_pool:
        if made >= N_VAL_NOREF:
            break
        o = idx[i]
        if not os.path.exists(o["p"]):
            continue
        ok, rec = verify(o["p"], o["t"])
        if not ok:
            continue
        row = make_row(o["t"], o["p"])
        row["duration"] = float(o["d"])
        val_rows.append(row)
        made += 1
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats.update({"total_train": len(rows), "val": len(val_rows),
                  "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(stats, open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print("S2_BUILD_DONE")


if __name__ == "__main__":
    main()
