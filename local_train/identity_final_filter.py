"""Точная арифметика identity-исключений + пересборка финального train.jsonl.

Проблема: identity_filter.py и identity_filter2.py писали в один файл по отдельности.
Здесь: единый проход с union-исключениями из ОРИГИНАЛА.
"""
import json
import os
import random
from collections import Counter

LOCAL = r"G:\AI\AuK\local_train"
IDD = os.path.join(LOCAL, "tests_packs", "identity_check")
S2 = os.path.join(LOCAL, "data_s2_full")
CORPUS = os.path.join(LOCAL, "corpus")
VER = os.path.join(S2, "v2_after_identity")

labels = {}
with open(os.path.join(IDD, "identity_gemini_labels.csv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            labels[parts[0]] = parts[1]

man = json.load(open(os.path.join(IDD, "manifest.json"), encoding="utf-8"))
man_by_id = {e["pair_id"]: e for e in man}

excluded_ids = [pid for pid, lab in labels.items() if lab in ("different", "uncertain")]
different_ids = [pid for pid, lab in labels.items() if lab == "different"]
suspicious_clusters = sorted({man_by_id[pid]["cluster"] for pid in different_ids})
all_excluded_clusters = sorted({man_by_id[pid]["cluster"] for pid in excluded_ids})
print(f"excluded pairs: {len(excluded_ids)} (different={len(different_ids)}, uncertain={len(excluded_ids)-len(different_ids)})")
print(f"suspicious clusters (from different): {suspicious_clusters}")
print(f"all clusters touched by exclusions: {len(all_excluded_clusters)}")

# pool и тренировочный набор
pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]
orig_train = os.path.join(S2, "train.jsonl")
rows = [json.loads(l) for l in open(orig_train, encoding="utf-8")]
train_pair_targets = {r["messages"][1]["content"][0]["audio_url"]
                      for r in rows if len(r["messages"][0]["content"]) > 1}

# 1) сопоставление 17 исключённых пар с тренировочным набором (по cluster+target_t+ref)
def norm(t):
    return t[:80].lower().strip()

found_pairs = []
for pid in excluded_ids:
    e = man_by_id[pid]
    cl, tt = e["cluster"], norm(e["tgt_text"])
    for p in pairs_all:
        if p["cluster"] == cl and norm(p["target_t"]) == tt:
            if p["target"] in train_pair_targets:
                found_pairs.append({"pair_id": pid, "cluster": cl,
                                    "target": p["target"], "in_train": True})
            else:
                found_pairs.append({"pair_id": pid, "cluster": cl,
                                    "target": p["target"], "in_train": False})
            break

found_in_train = [f for f in found_pairs if f["in_train"]]
print(f"\n17 pairs matched to pool: {len(found_pairs)} | found in train: {len(found_in_train)}")

# 2) пары из подозрительных кластеров в тренировочном наборе
cluster_pair_targets = {p["target"] for p in pairs_all if p["cluster"] in suspicious_clusters}
cluster_in_train = cluster_pair_targets & train_pair_targets
print(f"pairs from suspicious clusters in train: {len(cluster_in_train)}")

# 3) union исключение
exclude_targets = {f["target"] for f in found_in_train} | cluster_in_train
print(f"union excluded targets: {len(exclude_targets)}")

kept = []
for r in rows:
    if len(r["messages"][0]["content"]) > 1:
        if r["messages"][1]["content"][0]["audio_url"] in exclude_targets:
            continue
    kept.append(r)

n_noref = sum(1 for r in kept if len(r["messages"][0]["content"]) == 1)
n_pair = sum(1 for r in kept if len(r["messages"][0]["content"]) > 1)
print(f"\ntrain: {len(rows)} -> {len(kept)} (noref {n_noref}, pairs {n_pair})")
print(f"removed: {len(rows)-len(kept)}")

# 4) остаток пар подозрительных кластеров в финальном наборе
final_targets = {r["messages"][1]["content"][0]["audio_url"]
                 for r in kept if len(r["messages"][0]["content"]) > 1}
leftover = cluster_pair_targets & final_targets
print(f"pairs from suspicious clusters remaining: {len(leftover)} (должно быть 0)")

random.Random(7).shuffle(kept)
with open(os.path.join(VER, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

report = {
    "excluded_pairs_total": len(excluded_ids),
    "different": len(different_ids), "uncertain": len(excluded_ids) - len(different_ids),
    "suspicious_clusters_from_different": suspicious_clusters,
    "clusters_touched_total": all_excluded_clusters,
    "matched_from_17_in_train": len(found_in_train),
    "matched_but_not_in_train": len(found_pairs) - len(found_in_train),
    "suspicious_cluster_pairs_in_train": len(cluster_in_train),
    "union_excluded": len(exclude_targets),
    "train_before": len(rows), "train_after": len(kept),
    "noref": n_noref, "pairs": n_pair,
    "pairs_from_suspicious_clusters_remaining": len(leftover),
    "wording": "suspicious clusters excluded by automatic Gemini check (not human ground truth)",
}
json.dump(report, open(os.path.join(VER, "identity_final_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
print("IDENTITY_FINAL_DONE")
