"""Обработка разметки Gemini (по инструкции):
1) Исключить из пилота пары с вердиктом different/uncertain (сопоставление по исходным путям).
2) Проверить кластеры исключённых пар на смешение людей.
3) 7 пограничных «same» НЕ добавлять ниже порога 0.6.
4) Пересчитать финальные манифесты + пересечения, сохранить версии.
"""
import json
import os
import random
from collections import defaultdict

LOCAL = r"G:\AI\AuK\local_train"
IDD = os.path.join(LOCAL, "tests_packs", "identity_check")
S2 = os.path.join(LOCAL, "data_s2_full")
CORPUS = os.path.join(LOCAL, "corpus")

# --- загрузка разметки ---
labels = {}
with open(os.path.join(IDD, "identity_gemini_labels.csv"), encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            labels[parts[0]] = parts[1]
print(f"labels: {len(labels)}")
from collections import Counter
print("verdicts:", dict(Counter(labels.values())))

# --- манифест пар с исходными путями? ---
man = json.load(open(os.path.join(IDD, "manifest.json"), encoding="utf-8"))
# в манифесте ref_file/tgt_file — копии в паке; сопоставление к парам s2 по cluster+текстам
# надёжнее: матчить по pair_id -> (sim, cluster, ref_text, tgt_text) и потом найти пары в s2 по cluster+target_t
excluded_ids = [pid for pid, lab in labels.items() if lab in ("different", "uncertain")]
kept_same_ids = [pid for pid, lab in labels.items() if lab == "same"]
print(f"excluded pairs: {len(excluded_ids)} | kept 'same': {len(kept_same_ids)}")

# кластеры исключённых
excluded_clusters = defaultdict(list)
for e in man:
    if e["pair_id"] in excluded_ids:
        excluded_clusters[e["cluster"]].append(e["pair_id"])
print(f"clusters touched by exclusions: {len(excluded_clusters)}")
for cl, pids in sorted(excluded_clusters.items()):
    print(f"  cluster {cl}: {pids}")

# --- применить к s2_full: исключить пары, чьи (cluster, target_text) совпадают с excluded ---
def norm(t):
    return t[:80].lower().strip()

excl_keys = set()
for e in man:
    if e["pair_id"] in excluded_ids:
        excl_keys.add((e["cluster"], norm(e["tgt_text"])))
print(f"exclusion keys: {len(excl_keys)}")

rows = [json.loads(l) for l in open(os.path.join(S2, "train.jsonl"), encoding="utf-8")]
print(f"s2 train rows before: {len(rows)}")

kept_rows = []
removed = 0
removed_meta = []
for r in rows:
    is_pair = len(r["messages"][0]["content"]) > 1
    if is_pair:
        # cluster из meta? в s2_full meta нет cluster — но есть пара (ref_text? нет). Используем target текст + ref путь.
        tgt_text = r["messages"][1]["content"][0]["audio_url"]
        tgt_key = norm(r["messages"][0]["content"][0]["text"].split(": '")[-1])
        ref_path = r["messages"][0]["content"][1]["audio"]
        # найти пару в манускрипте по target пути
        found = None
        for e in man:
            # сравнение по target тексту (target_t) и кластеру невозможно без кластера в ряду;
            # но target путь в паре-строке s2 соответствует e? нет — e это копии в паке.
            pass
        kept_rows.append(r)
    else:
        kept_rows.append(r)

# Точный способ: сопоставить через s2_pairs_v2.jsonl по (cluster, target_t, ref)
pairs_all = [json.loads(l) for l in open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8")]
pair_by_target = {}
for p in pairs_all:
    pair_by_target.setdefault((p["cluster"], norms := p["target_t"][:80].lower().strip()), []).append(p)
# excluded pairs → их (cluster, target_t) → исключить из s2_full по target-пути
excl_targets = set()
for e in man:
    if e["pair_id"] in excluded_ids:
        cl = e["cluster"]
        tt = norm(e["tgt_text"])
        for p in pairs_all:
            if p["cluster"] == cl and norm(p["target_t"]) == tt:
                excl_targets.add(p["target"])
print(f"exact excluded target paths: {len(excl_targets)}")

kept_rows = []
removed = 0
for r in rows:
    is_pair = len(r["messages"][0]["content"]) > 1
    if is_pair:
        target = r["messages"][1]["content"][0]["audio_url"]
        if target in excl_targets:
            removed += 1
            continue
    kept_rows.append(r)

print(f"s2 train rows after: {len(kept_rows)} (removed {removed})")

# --- сохранить версии ---
ver = os.path.join(S2, "v2_after_identity")
os.makedirs(ver, exist_ok=True)
random.Random(7).shuffle(kept_rows)
with open(os.path.join(ver, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in kept_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
import shutil
shutil.copy(os.path.join(S2, "val.jsonl"), os.path.join(ver, "val.jsonl"))

# пересечения train/dev/final (по аудио-путям)
def paths_of(fn):
    s = set()
    for line in open(fn, encoding="utf-8"):
        o = json.loads(line)
        for msg in o["messages"]:
            for c in msg["content"]:
                if c.get("type") == "audio":
                    s.add(c.get("audio") or c.get("audio_url"))
    return s

tr = paths_of(os.path.join(ver, "train.jsonl"))
va = paths_of(os.path.join(ver, "val.jsonl"))
dev_fin = set()
with open(os.path.join(CORPUS, "split_manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["split"] in ("dev", "final"):
            dev_fin.add(o["p"])
print(f"train paths: {len(tr)}, val: {len(va)}")
print(f"train∩val: {len(tr & va)} | train∩dev/final: {len(tr & dev_fin)} | val∩dev/final: {len(va & dev_fin)}")

summary = {
    "labels": dict(Counter(labels.values())),
    "excluded_pairs": len(excluded_ids),
    "exact_excluded_targets": len(excl_targets),
    "s2_train_before": len(rows), "s2_train_after": len(kept_rows),
    "excluded_clusters": {str(k): v for k, v in excluded_clusters.items()},
    "border_same_not_added": [pid for pid in kept_same_ids if pid.startswith("border")],
    "train_val_overlap": len(tr & va), "train_devfin_overlap": len(tr & dev_fin),
}
json.dump(summary, open(os.path.join(ver, "identity_filter_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("IDENTITY_FILTER_DONE")
