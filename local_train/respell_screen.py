"""GigaAM screen for respell A/B: recall per variant + target word presence."""
import json
import os
import re
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train")
from gigaam_asr import transcribe  # noqa: E402

D = r"G:\AI\AuK\local_tests\respell_ab"
man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))


def norm(text):
    text = text.lower().replace("ё", "е").replace("+", "").replace("'", "").replace("-", "")
    return re.sub(r"[^а-я0-9\s]", " ", text).split()


rows = []
for m in man:
    hyp = transcribe(m["gen"])
    ref = norm(m["text"])
    got = norm(hyp)
    from collections import Counter
    rc, hc = Counter(ref), Counter(got)
    hits = sum((rc & hc).values())
    recall = hits / max(len(ref), 1)
    missing = sorted((rc - hc).elements())
    rows.append({"phrase": m["phrase"], "variant": m["variant"], "body": m["body"],
                 "file": os.path.basename(m["gen"]), "recall": round(recall, 3),
                 "missing": missing, "asr": hyp})
    print(f"p{m['phrase']} {m['variant']:<22} recall={recall:.2f} missing={missing[:4]}")
    sys.stdout.flush()

json.dump(rows, open(os.path.join(D, "screen_gigaam.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("SCREEN_DONE")
