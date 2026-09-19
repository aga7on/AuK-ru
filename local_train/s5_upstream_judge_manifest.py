"""Judge manifest для upstream-контроля s5: копия оригинальной структуры, пути -> variant dir.
Использование: python s5_upstream_judge_manifest.py [variant_dirname] [out_name]
Дефолт: s5_6000, s5_upstream_judge_manifest.json
"""
import json
import os
import sys

AUK = r"G:\AI\AuK"
SRC = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "upstream_judge_manifest.json")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "s5_6000"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "s5_upstream_judge_manifest.json"
    OUT = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", out_name)
    d = json.load(open(SRC, encoding="utf-8"))
    out = []
    for e in d:
        base = os.path.basename(e["file"])
        f = os.path.join(AUK, "local_tests", "upstream_control", variant, base)
        if not os.path.exists(f):
            continue
        e2 = dict(e)
        e2["id"] = variant + "__" + os.path.basename(base)[:-4]
        e2["file"] = f
        e2["wav"] = f
        out.append(e2)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
