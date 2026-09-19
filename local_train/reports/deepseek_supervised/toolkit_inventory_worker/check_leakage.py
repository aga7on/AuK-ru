"""Read-only leakage check: are eval_pack references present in the s1/s2 Russian training sets?

Streams train.jsonl/val.jsonl and records every audio path, then checks each eval_pack
(and phonetic_pack) `ref` against that set. Writes leakage_report.json. No writes elsewhere.
"""
from __future__ import annotations

import json
import os

AUK = r"G:\AI\AuK"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leakage_report.json")
TRAIN_SETS = [
    os.path.join(AUK, "local_train", "data", "train.jsonl"),
    os.path.join(AUK, "local_train", "data", "val.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_full", "train.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_full", "val.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_tools_v3", "train.jsonl"),
    os.path.join(AUK, "local_train", "data_s2_tools_v3", "val.jsonl"),
]
PACKS = [
    os.path.join(AUK, "local_tests", "eval_pack", "pack.json"),
    os.path.join(AUK, "local_tests", "phonetic_pack", "pack.json"),
]


def collect(path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in o.get("messages", []):
                for c in msg.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "audio":
                        p = c.get("audio") or c.get("audio_url")
                        if p:
                            seen.add(os.path.normcase(os.path.abspath(p)))
    return seen


def main() -> None:
    per_set: dict[str, set[str]] = {}
    for p in TRAIN_SETS:
        per_set[os.path.relpath(p, AUK).replace("\\", "/")] = collect(p)
    train = per_set.get("local_train/data_s2_full/train.jsonl", set())
    report = {"train_sets": {k: len(v) for k, v in per_set.items()}, "packs": {}}
    for pack in PACKS:
        if not os.path.isfile(pack):
            continue
        items = json.load(open(pack, encoding="utf-8"))
        rows = []
        for it in items:
            ref = it.get("ref")
            if not ref:
                continue
            key = os.path.normcase(os.path.abspath(ref))
            where = sorted(k for k, s in per_set.items() if key in s)
            rows.append({
                "id": it["id"], "kind": it.get("kind"), "ref": ref,
                "in_train": key in train, "seen_in_sets": where,
            })
        report["packs"][os.path.relpath(pack, AUK).replace("\\", "/")] = {
            "n_refs": len(rows),
            "n_in_train": sum(1 for r in rows if r["in_train"]),
            "n_seen_any": sum(1 for r in rows if r["seen_in_sets"]),
            "leaked_ids": [r["id"] for r in rows if r["seen_in_sets"]],
            "rows": rows,
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for pname, r in report["packs"].items():
        print(f"{pname}: {r['n_in_train']}/{r['n_refs']} refs in train; leaked={r['leaked_ids']}")


if __name__ == "__main__":
    main()
