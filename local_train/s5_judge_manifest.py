"""Judge manifest для s5 (120 контроль + 16 фонетика), судья v3 --manifest.
Использование: python s5_judge_manifest.py [tag] [control_dir] [phonetic_dir] [out_name]
Дефолт: s5, s5_control, s5_phonetic, s5_judge_manifest.json
"""
import json
import os
import re
import sys

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "s5"
    ctrl = sys.argv[2] if len(sys.argv) > 2 else "s5_control"
    phon = sys.argv[3] if len(sys.argv) > 3 else "s5_phonetic"
    out_name = sys.argv[4] if len(sys.argv) > 4 else "s5_judge_manifest.json"
    out = []
    for pack_path, run_dir in (
        (os.path.join(LT, "eval_pack", "pack.json"), os.path.join(LT, ctrl)),
        (os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, phon)),
    ):
        for e in json.load(open(pack_path, encoding="utf-8")):
            f = os.path.join(run_dir, e["id"] + ".wav")
            if not os.path.exists(f):
                continue
            kind = e.get("kind")
            if kind in ("tts", "phonetics"):
                mode, text = "tts", e.get("text") or ""
            elif kind == "clone":
                mode = "clone"
                m = re.search(r"'([^']*)'", e.get("instruction", ""))
                text = m.group(1) if m else ""
            else:
                mode, text = "tool", e["id"]
            out.append({"id": f"{tag}__{e['id']}", "mode": mode, "group": kind,
                        "instruction": e.get("instruction", ""), "text": text,
                        "goal": "", "checks": "", "ref": e.get("ref"), "file": f, "wav": f})
    p = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", out_name)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {p}")


if __name__ == "__main__":
    main()
