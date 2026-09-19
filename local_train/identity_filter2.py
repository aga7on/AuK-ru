"""Дополнение к identity_filter: исключить ВСЕ пары из подтверждённо смешанных кластеров.

«different» вердикт = кластер содержит минимум 2 разных человека → все пары кластера
непригодны для обучения клонированию.
"""
import json
import os
import random
import shutil
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
mixed_clusters = sorted({e["cluster"] for e in man if labels.get(e["pair_id"]) == "different"})
uncertain_clusters = sorted({e["cluster"] for e in man if labels.get(e["pair_id"]) == "uncertain"})
print(f"confirmed mixed clusters (different): {mixed_clusters}")
print(f"uncertain clusters: {len(uncertain_clusters)}")

# исходный набор и текущая версия
src_train = os.path.join(S2, "train.jsonl")
pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]

# target-пути всех пар из смешанных кластеров
mixed_targets = {p["target"] for p in pairs_all if p["cluster"] in mixed_clusters}
print(f"pairs from mixed clusters: {len(mixed_targets)}")

rows = [json.loads(l) for l in open(src_train, encoding="utf-8")]
kept = []
removed_mixed = 0
for r in rows:
    is_pair = len(r["messages"][0]["content"]) > 1
    if is_pair:
        target = r["messages"][1]["content"][0]["audio_url"]
        if target in mixed_targets:
            removed_mixed += 1
            continue
    kept.append(r)

print(f"train: {len(rows)} -> {len(kept)} (removed from mixed clusters: {removed_mixed})")

random.Random(7).shuffle(kept)
with open(os.path.join(VER, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# пересчёт состава
n_noref = sum(1 for r in kept if len(r["messages"][0]["content"]) == 1)
n_pair = sum(1 for r in kept if len(r["messages"][0]["content"]) > 1)
report = {
    "mixed_clusters_confirmed": mixed_clusters,
    "uncertain_clusters": uncertain_clusters,
    "pairs_from_mixed_clusters_removed": removed_mixed,
    "train_final": len(kept), "noref": n_noref, "pairs": n_pair,
    "border_same_not_added": [pid for pid, lab in labels.items()
                              if lab == "same" and pid.startswith("border")],
}
json.dump(report, open(os.path.join(VER, "identity_filter_report2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))

# пересечения
def paths_of(fn):
    s = set()
    for line in open(fn, encoding="utf-8"):
        o = json.loads(line)
        for msg in o["messages"]:
            for c in msg["content"]:
                if c.get("type") == "audio":
                    s.add(c.get("audio") or c.get("audio_url"))
    return s

tr = paths_of(os.path.join(VER, "train.jsonl"))
va = paths_of(os.path.join(VER, "val.jsonl"))
print(f"final: train={len(tr)} val={len(va)} overlap={len(tr & va)}")
print("IDENTITY_FILTER2_DONE")
