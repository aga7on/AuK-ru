"""Манифест судьи v3 для best-of-3 клонов s5@4500 (выбраны по WeSpeaker sim)."""
import json
import os
import re

AUK = r"G:\AI\AuK"
probe = json.load(open(rf"{AUK}\local_tests\tmp_seed_probe\seed_probe.json", encoding="utf-8"))
pack = json.load(open(rf"{AUK}\local_tests\eval_pack\pack.json", encoding="utf-8"))
by_id = {e["id"]: e for e in pack if e.get("kind") == "clone"}

out = []
for cid, sims in probe["per_clone"]:
    e = by_id[cid]
    best_seed = (7, 123, 999)[sims.index(max(sims))]
    f = rf"{AUK}\local_tests\tmp_seed_probe\{cid}_s{best_seed}.wav"
    m = re.search(r"'([^']*)'", e.get("instruction", ""))
    out.append({"id": f"s5_4500_best3__{cid}", "mode": "clone", "group": "clone",
                "instruction": e["instruction"], "text": m.group(1) if m else "",
                "goal": "Клон голоса референса на русском",
                "checks": "произношение чёткое, нейтральный русский акцент, сходство тембра, без артефактов",
                "ref": e.get("ref"), "file": f, "wav": f})
p = rf"{AUK}\local_train\reports\deepseek_supervised\s5_4500_best3_judge_manifest.json"
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"entries={len(out)} -> {p}")
