"""Speaker-disjoint tools val (по GATES §4.5 / STATUS 'Осталось' п.1).

Фильтрует val tools_v3 так, чтобы кластеры val НЕ пересекались ни с s2-речевым train
(v2_after_identity + B_mix_80_20), ни с tools train (B обучался на обоих).
Выход: data_s2_tools_v3/val_speaker_disjoint.jsonl + отчёт с инвариантами.
"""
import json
import os
import sys
from collections import Counter

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_train")
OUT_DIR = os.path.join(LT, "data_s2_tools_v3")
SPLIT_MANIFEST = os.path.join(LT, "corpus", "split_manifest.jsonl")


def cid(path):
    return os.path.basename(path or "").rsplit(".", 1)[0]


def load_split():
    m = {}
    for line in open(SPLIT_MANIFEST, encoding="utf-8"):
        d = json.loads(line)
        m[cid(d["p"])] = (int(d["cluster"]), d.get("split"))
    return m


def audio_ids(path):
    ids = set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        for msg in d["messages"]:
            for c in msg["content"]:
                if c.get("type") == "audio":
                    ids.add(cid(c.get("audio") or c.get("audio_url")))
    return ids


def tools_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        rows.append(d)
    return rows


def row_cluster(row, split):
    meta = row.get("meta", {})
    cl = meta.get("speaker_cluster")
    if cl is not None:
        return int(cl)
    src = row["messages"][0]["content"][1]["audio"] if len(
        row["messages"][0]["content"]) > 1 else None
    return split.get(cid(src), (None, None))[0]


def main():
    split = load_split()

    # train clusters модели: речь (v2 + B_mix) + tools train
    speech_train_ids = audio_ids(os.path.join(LT, "data_s2_full", "v2_after_identity", "train.jsonl"))
    bmix_ids = audio_ids(os.path.join(LT, "data_s2_full", "B_mix_80_20", "train.jsonl"))
    s2_train_ids = speech_train_ids | bmix_ids
    s2_train_clusters = {split[i][0] for i in s2_train_ids if i in split and split[i][0] is not None}

    tools_train = tools_rows(os.path.join(OUT_DIR, "train.jsonl"))
    tools_train_clusters = {row_cluster(r, split) for r in tools_train} - {None}

    blocked = s2_train_clusters | tools_train_clusters

    val_rows = tools_rows(os.path.join(OUT_DIR, "val.jsonl"))
    kept, dropped = [], 0
    drop_clusters = set()
    for r in val_rows:
        cl = row_cluster(r, split)
        if cl is None or cl in blocked:
            dropped += 1
            if cl is not None:
                drop_clusters.add(cl)
        else:
            kept.append(r)

    out_path = os.path.join(OUT_DIR, "val_speaker_disjoint.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ИНВАРИАНТЫ (две независимые проверки)
    kept_ids = set()
    kept_clusters = set()
    for r in kept:
        kept_ids.add(cid(r["messages"][0]["content"][1]["audio"]))
        kept_clusters.add(row_cluster(r, split))
    inv_overlap_ids = kept_ids & s2_train_ids
    inv_overlap_clusters = kept_clusters & blocked
    train_ids = {cid(r["messages"][0]["content"][1]["audio"]) for r in tools_train}
    inv_overlap_train_ids = kept_ids & train_ids

    ops = Counter(r["meta"]["op"] for r in kept)
    dur = sum(r.get("duration", 0) for r in kept)

    report = {
        "input_val_rows": len(val_rows), "kept_rows": len(kept), "dropped_rows": dropped,
        "dropped_unique_clusters": len(drop_clusters),
        "kept_unique_source_ids": len(kept_ids), "kept_unique_clusters": len(kept_clusters),
        "inv_overlap_ids_vs_s2_train": len(inv_overlap_ids),
        "inv_overlap_clusters_vs_blocked": len(inv_overlap_clusters),
        "inv_overlap_ids_vs_tools_train": len(inv_overlap_train_ids),
        "ops": dict(ops), "total_hours": round(dur / 3600, 2),
        "blocked_clusters_total": len(blocked),
    }
    out_json = os.path.join(OUT_DIR, "val_speaker_disjoint_report.json")
    json.dump(report, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    ok = (not inv_overlap_ids and not inv_overlap_clusters and not inv_overlap_train_ids
          and len(kept) >= 400)
    print("INVARIANTS_OK" if ok else "INVARIANTS_FAIL")


if __name__ == "__main__":
    main()
