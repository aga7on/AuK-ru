"""Какие пары кластеров 8681/8720/9220 остались в финальном train."""
import json
import os

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
VER = os.path.join(LOCAL, "data_s2_full", "v2_after_identity")

pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]
final_rows = [json.loads(l) for l in open(os.path.join(VER, "train.jsonl"), encoding="utf-8")]
final_targets = {r["messages"][1]["content"][0]["audio_url"]
                 for r in final_rows if len(r["messages"][0]["content"]) > 1}

for cl in (8681, 8720, 9220):
    cl_targets = {p["target"] for p in pairs_all if p["cluster"] == cl}
    in_final = cl_targets & final_targets
    print(f"cluster {cl}: pool pairs={len(cl_targets)}, in final train={len(in_final)}")
    for t in in_final:
        for p in pairs_all:
            if p["target"] == t:
                print(f"   {os.path.basename(t)} | ref_d={p['ref_d']:.1f} tgt_d={p['target_d']:.1f} | {p['target_t'][:60]}")
