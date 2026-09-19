"""Summarize razvitie A/B results: per file show rep/seed, recall, whether 'развитие' survived."""
import glob
import json
import os

ROOT = r"G:\AI\AuK\local_tests\razvitie_ab"


def check(asr):
    a = asr.lower().replace("ё", "е")
    if "развит" in a:
        return "OK(развитие)"
    if "разит" in a:
        return "MANGLED(разитие)"
    return "OTHER"


for tag in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, tag)
    rp = os.path.join(d, "results.json")
    if not os.path.isfile(rp):
        continue
    man = {m["gen"]: m for m in json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))}
    res = json.load(open(rp, encoding="utf-8"))
    print(f"== {tag}")
    for r in sorted(res, key=lambda x: x["gen"]):
        m = man[r["gen"]]
        print("  %-9s s%-4s recall=%.2f missing=%s %s | %s" % (
            m["representation"], m["seed"], r.get("recall", 0), str(r.get("missing_words", []))[:40],
            check(r.get("asr", "")), (r.get("asr") or "")[:70]))
