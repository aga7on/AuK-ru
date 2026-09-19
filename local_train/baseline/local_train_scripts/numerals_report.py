"""Report for numerals/lingo A/B: variant x seed with judge metrics + ASR."""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else r"G:\AI\AuK\local_tests\numerals_ab"


def main():
    man = {m["gen"]: m for m in json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))}
    res = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
    for r in sorted(res, key=lambda x: x["gen"]):
        m = man[r["gen"]]
        j = r.get("judge") or {}
        name = m.get("variant", m.get("setting", "?"))
        if "phrase" in m:
            name = "p%s/%s" % (m["phrase"], name)
        print("%-22s s%-5s ovrl=%-3s prosody=%-3s nat=%-3s recall=%.2f missing=%s | %s" % (
            name, m["seed"], j.get("overall"), j.get("prosody"), j.get("naturalness"),
            r.get("recall_fixed", r.get("recall", 0)), str(r.get("missing_fixed", r.get("missing_words", [])))[:45],
            (r.get("asr") or "")[:70]))


if __name__ == "__main__":
    main()
