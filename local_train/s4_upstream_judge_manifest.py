"""Judge manifest для upstream-контроля s4@4000: копия оригинальной структуры, пути -> s4_4000."""
import json
import os

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "upstream_judge_manifest.json")
OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "s4_upstream_judge_manifest.json")


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    out = []
    for e in d:
        base = os.path.basename(e["file"])
        f = os.path.join(AUK, "local_tests", "upstream_control", "s4_4000", base)
        if not os.path.exists(f):
            continue
        e2 = dict(e)
        e2["id"] = "s4_4000__" + os.path.basename(base)[:-4]
        e2["file"] = f
        e2["wav"] = f
        out.append(e2)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
