import json
import os
import re

TRAIN = r"G:\AI\_tmp\train_s1.log"
CTRL = r"G:\AI\AuK\local_train\run_ru_s1\control.jsonl"

d = open(TRAIN, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
lines = [l.rstrip() for l in d.splitlines() if l.strip()]
last_start = max([i for i, l in enumerate(lines) if "ATTEMPT" in l and "START" in l])
seg = lines[last_start:]
ups = [m for m in (re.search(r"update (\d+)/(\d+)\] loss=([\d.]+)", l) for l in seg) if m]
if ups:
    print("train: u%s/%s loss=%s | logged steps: %d | OOM: %s" % (
        ups[-1].group(1), ups[-1].group(2), ups[-1].group(3), len(ups),
        any("OutOfMemory" in l for l in seg)))
    print("last losses:", [(m.group(1), m.group(3)) for m in ups[-4:]])
else:
    print("train: no updates in current attempt | OOM:", any("OutOfMemory" in l for l in seg))
exit_lines = [l for l in seg if "EXIT" in l]
if exit_lines:
    print("EXIT:", exit_lines[-1][:80])

if os.path.exists(CTRL):
    rows = [json.loads(l) for l in open(CTRL, encoding="utf-8")]
    print("controller rows:", len(rows))
    for r in rows[-3:]:
        m = r["means"]
        print("  u%s [%s] score=%s ovrl=%.1f accent=%.1f palat=%s nat=%.1f sim=%.1f text=%s/%s" % (
            r["update"], r["label"], r["score"], m["overall"], m["accent"],
            (round(m["palatalization"], 1) if m.get("palatalization") is not None else None),
            m["naturalness"], m["voice_similarity"], m["text_matches"], r.get("n_prompts", "?")))
else:
    print("controller: no rows yet")

import glob
revs = sorted(glob.glob(r"G:\AI\AuK\local_train\user_reviews\*.json"))
if revs:
    print("user reviews:")
    for p in revs[-5:]:
        try:
            r = json.load(open(p, encoding="utf-8"))
            v = r.get("verdict_json", {})
            print("  %s overall=%s verdict=%s issues=%s" % (
                os.path.basename(p), v.get("overall"), v.get("verdict"), (v.get("issues") or "")[:60]))
        except Exception:
            pass
