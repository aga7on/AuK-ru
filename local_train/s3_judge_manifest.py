"""Judge manifest для s3@2000 (120 контроль + 16 фонетика), судья v3 --manifest."""
import json
import os
import re

AUK = r"G:\AI\AuK"
LT = os.path.join(AUK, "local_tests")


def main():
    out = []
    for pack_path, run_dir in (
        (os.path.join(LT, "eval_pack", "pack.json"), os.path.join(LT, "s3_control")),
        (os.path.join(LT, "phonetic_pack", "pack.json"), os.path.join(LT, "s3_phonetic")),
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
            out.append({"id": f"s3__{e['id']}", "mode": mode, "group": kind,
                        "instruction": e.get("instruction", ""), "text": text,
                        "goal": "", "checks": "", "ref": e.get("ref"), "file": f, "wav": f})
    p = os.path.join(AUK, "local_train", "reports", "deepseek_supervised", "s3_judge_manifest.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"entries={len(out)} -> {p}")


if __name__ == "__main__":
    main()
