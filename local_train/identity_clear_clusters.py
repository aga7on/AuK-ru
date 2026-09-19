"""Финальная очистка: все пары кластеров 8681/8720/9220 удалить из train и val.

Отчёт: matched из 17 пар, удаления по трём кластерам, остаток = 0.
Wording: suspicious clusters excluded by automatic check (Gemini), not human ground truth.
"""
import json
import os
import random

LOCAL = r"G:\AI\AuK\local_train"
S2 = os.path.join(LOCAL, "data_s2_full")
CORPUS = os.path.join(LOCAL, "corpus")
VER = os.path.join(S2, "v2_after_identity")
CLEAR_CLUSTERS = {8681, 8720, 9220}

pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]
clear_targets = {p["target"] for p in pairs_all if p["cluster"] in CLEAR_CLUSTERS}
print(f"targets to clear: {len(clear_targets)} (clusters {sorted(CLEAR_CLUSTERS)})")

report = {"clusters_cleared": sorted(CLEAR_CLUSTERS), "by_cluster": {}}
for cl in sorted(CLEAR_CLUSTERS):
    cl_t = {p["target"] for p in pairs_all if p["cluster"] == cl}
    report["by_cluster"][str(cl)] = {"pool_pairs": len(cl_t)}

for fn in ("train.jsonl", "val.jsonl"):
    p = os.path.join(VER, fn)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    bef = len(rows)
    kept = []
    removed_here = 0
    for r in rows:
        if len(r["messages"][0]["content"]) > 1:
            target = r["messages"][1]["content"][0]["audio_url"]
            if target in clear_targets:
                removed_here += 1
                continue
        kept.append(r)
    if fn == "train.jsonl":
        random.Random(7).shuffle(kept)
    with open(p, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{fn}: {bef} -> {len(kept)} (removed {removed_here})")
    report[fn] = {"before": bef, "after": len(kept), "removed": removed_here}

# по-кластерная статистика удалений (по train до очистки — считаем заново от оригинала)
orig = [json.loads(l) for l in open(os.path.join(S2, "train.jsonl"), encoding="utf-8")]
orig_pair_targets = {r["messages"][1]["content"][0]["audio_url"]
                     for r in orig if len(r["messages"][0]["content"]) > 1}
for cl in sorted(CLEAR_CLUSTERS):
    cl_t = {p["target"] for p in pairs_all if p["cluster"] == cl}
    in_orig = cl_t & orig_pair_targets
    report["by_cluster"][str(cl)]["in_train_orig"] = len(in_orig)

# финальный состав
final = [json.loads(l) for l in open(os.path.join(VER, "train.jsonl"), encoding="utf-8")]
n_noref = sum(1 for r in final if len(r["messages"][0]["content"]) == 1)
n_pair = sum(1 for r in final if len(r["messages"][0]["content"]) > 1)
report["train_final"] = {"noref": n_noref, "pairs": n_pair, "total": len(final)}

# остаток кластеров = 0?
final_pair_targets = {r["messages"][1]["content"][0]["audio_url"]
                      for r in final if len(r["messages"][0]["content"]) > 1}
leftover = clear_targets & final_pair_targets
report["pairs_remaining_from_cleared_clusters"] = len(leftover)

report["matched_from_17_in_train"] = 1  # conf_00 (cluster 9220, request_failed -> no result)
report["wording"] = "suspicious clusters excluded by automatic Gemini check (not human ground truth)"

json.dump(report, open(os.path.join(VER, "identity_final_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
print("FINAL_CLEAR_DONE")
