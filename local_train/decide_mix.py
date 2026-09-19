"""Decide the stage-1 text representation mix from the micro-experiment results."""
import json
import sys


def mean_accent(path):
    rows = json.load(open(path, encoding="utf-8"))
    vals = []
    for r in rows:
        j = r.get("judge") or {}
        v = j.get("accent")
        if v is None:
            v = j.get("naturalness", 0)
        vals.append(float(v))
    return (sum(vals) / max(len(vals), 1)), len(vals)


def main():
    cyr_path, tr_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    a_cyr, n1 = mean_accent(cyr_path)
    a_tr, n2 = mean_accent(tr_path)
    if a_cyr >= a_tr - 0.3:
        mix = "cyr_stress:0.65,cyr_plain:0.25,translit:0.10"
        verdict = "cyrillic ok"
    else:
        mix = "cyr_stress:0.40,cyr_plain:0.20,translit:0.40"
        verdict = "cyrillic weaker -> translit-heavy mix"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(mix)
    print(f"microtest: accent cyrillic={a_cyr:.2f} ({n1}) vs translit={a_tr:.2f} ({n2}) -> {verdict}: {mix}")


if __name__ == "__main__":
    main()
