import json
r = json.load(open(r"G:\AI\AuK\local_train\data_s2_full\build_report.json", encoding="utf-8"))
ok = {k: v for k, v in r.items() if k.startswith("ok_")}
drops = {k: v for k, v in r.items() if k.startswith("drop_")}
print("ACCEPTED:", json.dumps(ok, ensure_ascii=False))
print("DROPS total:", sum(drops.values()))
top = sorted(drops.items(), key=lambda x: -x[1])[:8]
print("TOP DROPS:", json.dumps(top, ensure_ascii=False))
vf = r"G:\AI\AuK\local_train\data_s2_full\val.jsonl"
n = sum(1 for _ in open(vf, encoding="utf-8"))
print(f"val.jsonl rows: {n}")
