"""Точная раскладка удалённых пар по кластерам."""
import json
import os

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

print("=== Все 17 исключённых пар: кластеры ===")
for pid in sorted(excluded_ids):
    e = man_by_id[pid]
    print(f"  {pid}: cluster={e['cluster']} sim={e['sim']} label={labels[pid]}")

# находим пары в тренировочном наборе (по cluster+target_t)
pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]
rows = [json.loads(l) for l in open(os.path.join(S2, "train.jsonl"), encoding="utf-8")]
train_pair_targets = {r["messages"][1]["content"][0]["audio_url"]
                      for r in rows if len(r["messages"][0]["content"]) > 1}

def norm(t):
    return t[:80].lower().strip()

print("\n=== 17 пар: найдено в train ===")
removed_by_cluster = {}
for pid in excluded_ids:
    e = man_by_id[pid]
    cl, tt = e["cluster"], norm(e["tgt_text"])
    for p in pairs_all:
        if p["cluster"] == cl and norm(p["target_t"]) == tt:
            in_train = p["target"] in train_pair_targets
            if in_train:
                removed_by_cluster.setdefault(cl, []).append(pid)
                print(f"  {pid} (cluster {cl}) -> В TRAIN, target={os.path.basename(p['target'])}")
            break

print("\n=== Пары подозрительных кластеров в train ===")
suspicious = sorted({man_by_id[pid]["cluster"] for pid in different_ids})
for cl in suspicious:
    cl_targets = {p["target"] for p in pairs_all if p["cluster"] == cl}
    in_train = cl_targets & train_pair_targets
    print(f"  cluster {cl}: {len(in_train)} пар в train")
    for t in in_train:
        print(f"    {os.path.basename(t)}")

# union removed targets
excl_from_17 = set()
for pid in excluded_ids:
    e = man_by_id[pid]
    cl, tt = e["cluster"], norm(e["tgt_text"])
    for p in pairs_all:
        if p["cluster"] == cl and norm(p["target_t"]) == tt and p["target"] in train_pair_targets:
            excl_from_17.add(p["target"])
            break
excl_from_clusters = set()
for cl in suspicious:
    cl_targets = {p["target"] for p in pairs_all if p["cluster"] == cl}
    excl_from_clusters |= (cl_targets & train_pair_targets)

union = excl_from_17 | excl_from_clusters
print(f"\nиз 17-пар исключено: {len(excl_from_17)}")
print(f"из подозрительных кластеров: {len(excl_from_clusters)}")
print(f"union: {len(union)}")

# какие кластеры затронуты удалениями
all_clusters_removed = set()
for pid in excluded_ids:
    e = man_by_id[pid]
    cl, tt = e["cluster"], norm(e["tgt_text"])
    for p in pairs_all:
        if p["cluster"] == cl and norm(p["target_t"]) == tt and p["target"] in train_pair_targets:
            all_clusters_removed.add(cl)
            break
for cl in suspicious:
    if {p["target"] for p in pairs_all if p["cluster"] == cl} & train_pair_targets:
        all_clusters_removed.add(cl)
print(f"кластеры, затронутые удалениями: {sorted(all_clusters_removed)}")

# остаток
final_targets = {r["messages"][1]["content"][0]["audio_url"]
                 for r in [json.loads(l) for l in open(os.path.join(VER, "train.jsonl"), encoding="utf-8")]
                 if len(r["messages"][0]["content"]) > 1}
remaining = set()
for cl in all_clusters_removed:
    cl_targets = {p["target"] for p in pairs_all if p["cluster"] == cl}
    remaining |= (cl_targets & final_targets)
print(f"пары этих кластеров, оставшиеся в финальном train: {len(remaining)} (должно быть 0)")
