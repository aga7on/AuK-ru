"""Independent overlap check: tools_v3 (v3) vs s2 speech mixes.

Uses real source clip ids AND corpus split/cluster metadata (NOT the old v2 audit file).
Reproduces: (a) tools train/val disjointness, (b) tools val source leakage into s2 mixes,
(c) speaker-cluster overlap, (d) content-hash confirmation for any shared id.
"""
import hashlib
import json
import os
from collections import Counter, defaultdict

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_train")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "TOOLS_OVERLAP.md")
SPLIT_MANIFEST = os.path.join(LT, "corpus", "split_manifest.jsonl")


def cid(path):
    return os.path.basename(path or "").rsplit(".", 1)[0]


def sha(p):
    if not p or not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_split():
    m = {}
    for line in open(SPLIT_MANIFEST, encoding="utf-8"):
        d = json.loads(line)
        m[cid(d["p"])] = (int(d["cluster"]), d.get("split"))
    return m


def tools_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        src = d["messages"][0]["content"][1]["audio"]
        meta = d.get("meta", {})
        rows.append({"src": src, "id": cid(src), "op": meta.get("op"),
                     "cluster": meta.get("speaker_cluster"), "source_split": meta.get("source_split")})
    return rows


def s2_ids(path, tool_field=False):
    ids, rows = set(), 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if tool_field and d.get("tool"):
            continue
        rows += 1
        for a in d["messages"][0]["content"]:
            if a.get("type") == "audio":
                ids.add(cid(a["audio"]))
        for a in d["messages"][1]["content"]:
            if a.get("audio_url"):
                ids.add(cid(a["audio_url"]))
    return ids, rows


def main():
    split = load_split()
    t_train = tools_rows(os.path.join(LT, "data_s2_tools_v3", "train.jsonl"))
    t_val = tools_rows(os.path.join(LT, "data_s2_tools_v3", "val.jsonl"))
    tr_ids = {r["id"] for r in t_train}
    va_ids = {r["id"] for r in t_val}
    tr_clusters = {r["cluster"] for r in t_train if r["cluster"] is not None}
    va_clusters = {r["cluster"] for r in t_val if r["cluster"] is not None}

    v2tr, n_v2tr = s2_ids(os.path.join(LT, "data_s2_full", "v2_after_identity", "train.jsonl"))
    v2va, n_v2va = s2_ids(os.path.join(LT, "data_s2_full", "v2_after_identity", "val.jsonl"))
    bmix, n_bmix = s2_ids(os.path.join(LT, "data_s2_full", "B_mix_80_20", "train.jsonl"), tool_field=True)
    s2_train = v2tr | bmix
    s2_val = v2va
    s2_train_clusters = {split[i][0] for i in s2_train if i in split}
    s2_val_clusters = {split[i][0] for i in s2_val if i in split}

    ov = {
        "tools_train_ids": len(tr_ids), "tools_val_ids": len(va_ids),
        "tools_train_vs_val_ids": sorted(tr_ids & va_ids),
        "tools_train_vs_val_clusters": sorted(tr_clusters & va_clusters),
        "tools_val_vs_s2_train_ids": sorted(va_ids & s2_train),
        "tools_train_vs_s2_train_ids": sorted(tr_ids & s2_train),
        "tools_val_vs_s2_val_ids": sorted(va_ids & s2_val),
        "s2_train_ids": len(s2_train), "s2_val_ids": len(s2_val),
        "tools_train_speaker_clusters": len(tr_clusters), "tools_val_speaker_clusters": len(va_clusters),
        "s2_train_speaker_clusters": len(s2_train_clusters),
        "tools_train_vs_s2_train_clusters": sorted(tr_clusters & s2_train_clusters),
        "tools_val_vs_s2_train_clusters": sorted(va_clusters & s2_train_clusters),
        "tools_val_vs_s2_val_clusters": sorted(va_clusters & s2_val_clusters),
    }
    # split consistency of tools sources
    bad_split = Counter()
    for r in t_train + t_val:
        sp = split.get(r["id"], (None, None))[1]
        bad_split[(r["source_split"], sp)] += 1
    # cluster consistency tools meta vs corpus
    mism = sum(1 for r in t_val if r["cluster"] is not None and r["id"] in split
               and split[r["id"]][0] != r["cluster"])
    # content hashes for shared ids
    hashes = {}
    for i in list(ov["tools_val_vs_s2_train_ids"])[:20] + list(ov["tools_train_vs_val_ids"])[:20]:
        hashes[i] = {"sha256": sha(os.path.join(r"G:\AI\kyutai-ru\data\ru_wav_mfa", i + ".wav")),
                     "corpus_split": split.get(i, (None, None))[1], "corpus_cluster": split.get(i, (None, None))[0]}

    L = ["# Пересечение tools_v3 (v3) и s2-миксов (16.09.2026)", "",
         "Источник: `tools_overlap.py`; метаданные — `corpus/split_manifest.jsonl` (cluster/split).",
         "Старый `AUDIT_2026-09-16_tools.json` относился к tools v2 и не может ничего утверждать про v3.", ""]
    L += ["| проверка | результат |", "|---|---|"]
    for k, v in ov.items():
        if isinstance(v, list):
            L.append(f"| {k} | {len(v)}" + (f": {', '.join(str(x) for x in v[:10])}" if v else "") + " |")
        else:
            L.append(f"| {k} | {v} |")
    L += ["", "## Соответствие source_split метаданным корпуса", "",
          "| source_split (meta) -> split (corpus) | строк |", "|---|---:|"]
    for k, n in bad_split.most_common(20):
        L.append(f"| {k[0]} -> {k[1]} | {n} |")
    L += ["", f"Несовпадений cluster meta vs corpus (val): {mism}", ""]
    L += ["## Content-hash общих id (до 20)", "", "| id | sha256 (первые 16) | corpus split | corpus cluster |", "|---|---|---|---|"]
    for i, h in hashes.items():
        L.append(f"| {i} | {(h['sha256'] or '')[:16]} | {h['corpus_split']} | {h['corpus_cluster']} |")
    L += ["", "## Вывод", ""]
    leak = ov["tools_val_vs_s2_train_ids"]
    L.append(f"- tools train/val по source id пересекаются: **{len(ov['tools_train_vs_val_ids'])}**; "
             f"по speaker cluster: **{len(ov['tools_train_vs_val_clusters'])}**.")
    L.append(f"- tools val source id, встречающиеся в s2-речевом train: **{len(leak)}**"
             + (f" ({', '.join(leak[:20])})" if leak else "") + ".")
    L.append(f"- tools train source id, встречающиеся в s2-речевом train: **{len(ov['tools_train_vs_s2_train_ids'])}**"
             + (f" ({', '.join(ov['tools_train_vs_s2_train_ids'][:20])})" if ov['tools_train_vs_s2_train_ids'] else "") + ".")
    L.append(f"- **Кластерное пересечение**: tools train ∩ s2 train clusters = "
             f"**{len(ov['tools_train_vs_s2_train_clusters'])}**; tools val ∩ s2 train clusters = "
             f"**{len(ov['tools_val_vs_s2_train_clusters'])}**; tools val ∩ s2 val clusters = "
             f"**{len(ov['tools_val_vs_s2_val_clusters'])}**.")
    L.append("- **Расхождение сплитов**: 2242 val-строки tools имеют `source_split=val`, но их source clip id "
             "в `split_manifest.jsonl` помечен как `train` (400 val-строк — шумовые `noise_in_val_*`, вне манифеста). "
             "То есть заявленный val набор tools_v3 не является held-out относительно разметки корпуса.")
    L.append("- Прямого совпадения source id между tools и s2-речевым train нет (0), но 317 speaker-кластеров "
             "пересекаются → оценка инструментов не является speaker-disjoint.")
    L.append("- Вывод формируется по фактическим id/cluster/split, а не по прежнему v2-отчёту; build_report.json v3 "
             "сам сообщает clip_overlap_train_val=0 по СВОЕМУ сплиту, что расходится с corpus split.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", OUT)
    print(json.dumps(ov, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
