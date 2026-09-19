"""Report for stress variants: judge stress field + issues + ASR text per variant."""
import json
import os

D = r"G:\AI\AuK\local_tests\stress_variants"


def main():
    man = {m["gen"]: m for m in json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))}
    res = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
    for r in res:
        m = man[r["gen"]]
        j = r.get("judge") or {}
        print("%-20s stress=%-4s overall=%-3s issues=%s | asr=%s" % (
            m["variant"], j.get("stress"), j.get("overall"),
            (j.get("issues") or "")[:80], (r.get("asr") or "")[:60]))


if __name__ == "__main__":
    main()
