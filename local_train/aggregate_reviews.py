"""Aggregate all human/Qwen-3.5-Omni reviews into a ranked scorecard."""
import glob
import json
import os

REV = r"G:\AI\AuK\local_train\user_reviews"


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(REV, "*.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for key in ("verdict_json", "verdict_json_v2"):
            v = r.get(key)
            if not v:
                continue
            rows.append({
                "sample": r.get("sample"),
                "overall": v.get("overall"),
                "verdict": v.get("verdict"),
                "text": v.get("text_fidelity"),
                "endings": v.get("endings"),
                "nat": v.get("naturalness"),
                "prosody": v.get("prosody"),
                "stress": v.get("stress"),
                "palat": v.get("palatalization"),
                "artifacts": v.get("artifacts"),
                "issues": (v.get("issues") or "")[:70],
            })
    rows.sort(key=lambda x: -(x["overall"] or 0))
    print("sample | overall | verdict | text/end/nat/pros/stress/palat | issues")
    for x in rows:
        print("%-22s | %-3s | %-10s | %s/%s/%s/%s/%s/%s | %s" % (
            x["sample"], x["overall"], str(x["verdict"])[:10],
            x["text"], x["endings"], x["nat"], x["prosody"], x["stress"], x["palat"], x["issues"]))
    print("total reviews:", len(rows))


if __name__ == "__main__":
    main()
