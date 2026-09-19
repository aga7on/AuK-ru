"""Build listening indexes for stage-1 samples and controller eval sets."""
import glob
import json
import os

S = r"G:\AI\AuK\local_train\run_ru_s1\samples"
val0 = json.loads(open(r"G:\AI\AuK\local_train\data\val.jsonl", encoding="utf-8").readline())
vtext = val0["messages"][0]["content"][0]["text"]

metrics = {}
mp = r"G:\AI\AuK\local_train\run_ru_s1\eval_samples.jsonl"
if os.path.exists(mp):
    for line in open(mp, encoding="utf-8"):
        try:
            r = json.loads(line)
            metrics[r["update"]] = r
        except Exception:
            pass

lines = ["VALIDATION TEXT (которая звучит во всех семплах):", vtext, "",
         "update | watcher: gen ovrl/f0 vs tgt ovrl/f0 | dur ratio"]
gens = sorted(glob.glob(os.path.join(S, "update_*_gen.wav")), key=lambda p: int(p.split("_")[-2]))
for g in gens:
    u = int(g.split("_")[-2])
    m = metrics.get(u)
    if m:
        gm, tm = m["gen_metrics"], m["tgt_metrics"]
        lines.append("u%-6d | gen %s/%s vs tgt %s/%s | ratio %s" % (
            u, gm["ovrl"], gm["f0"], tm["ovrl"], tm["f0"], m["dur_ratio"]))
    else:
        lines.append("u%-6d | (no watcher metrics)" % u)
open(os.path.join(S, "LISTEN.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("LISTEN.txt written:", len(gens), "samples")

lines2 = ["CONTROLLER EVAL SETS (8 фраз; gen_i.wav <-> текст из manifest.json):"]
for d in sorted(glob.glob(r"G:\AI\AuK\local_train\run_ru_s1\evals\lora_u*"), key=lambda p: int(p.split("u")[-1])):
    lines2.append("== " + os.path.basename(d))
    man = os.path.join(d, "manifest.json")
    if os.path.exists(man):
        for m in json.load(open(man, encoding="utf-8")):
            lines2.append("   %s | %s" % (m["gen"], m["text"]))
open(r"G:\AI\AuK\local_train\run_ru_s1\evals\INDEX.txt", "w", encoding="utf-8").write("\n".join(lines2))
print("INDEX.txt written")
