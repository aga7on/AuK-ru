"""Сводка Dev-набора (шаг 3): метрики + судья по категориям и голосам."""
import json
import os
from collections import defaultdict

BASE = r"G:\AI\AuK\local_tests\dev_set\u18000_prod"
m = json.load(open(os.path.join(BASE, "ru_metrics.json"), encoding="utf-8"))
g = json.load(open(os.path.join(BASE, "results_gemini.json"), encoding="utf-8"))
judge = {r["file_name"]: (r.get("judge") or {}) for r in g}

rows = []
for r in m:
    f = r["file"]
    j = judge.get(f, {})
    # voice/cat из имени devpXX_cat_voice.wav (cat может содержать подчёркивание)
    stem = f.replace(".wav", "")
    body = stem.split("_", 1)[1] if "_" in stem else stem
    tokens = body.split("_")
    if not tokens:
        cat, voice = "?", "?"
    elif len(tokens) == 1:
        cat, voice = tokens[0], "?"
    else:
        voice = "_".join(tokens[-2:]) if tokens[-2] in ("alt", "nat", "user") else tokens[-1]
        cat = body[: len(body) - len(voice) - 1]
    rows.append({"file": f, "cat": cat, "voice": voice,
                 "wer": r["text_metrics"]["wer"], "verdict": r["verdict"],
                 "overall": j.get("overall"), "mangled": j.get("words_mangled", []),
                 "issues": (j.get("issues") or "")[:80], "heard": (r.get("heard") or "")[:70]})

by_cat = defaultdict(list)
by_voice = defaultdict(list)
for r in rows:
    by_cat[r["cat"]].append(r)
    by_voice[r["voice"]].append(r)


def agg(rs):
    ov = [r["overall"] for r in rs if isinstance(r["overall"], (int, float))]
    wers = [r["wer"] for r in rs]
    clean = sum(1 for w in wers if w <= 0.15)
    return len(rs), round(sum(wers) / max(len(wers), 1), 3), clean, \
        round(sum(ov) / max(len(ov), 1), 2), (min(ov) if ov else "?")


lines = ["# DEV SUMMARY — u18000, продуктовый режим (48 файлов, 4 голоса, seed 1234)", ""]
lines.append("| Категория | n | ср.WER | чистых | ср.судья | мин |")
lines.append("|---|---|---|---|---|---|")
for cat, rs in sorted(by_cat.items()):
    n, w, c, o, mn = agg(rs)
    lines.append(f"| {cat} | {n} | {w} | {c}/{n} | {o} | {mn} |")
lines.append("")
lines.append("| Голос | n | ср.WER | чистых | ср.судья | мин |")
lines.append("|---|---|---|---|---|---|")
for v, rs in sorted(by_voice.items()):
    n, w, c, o, mn = agg(rs)
    lines.append(f"| {v} | {n} | {w} | {c}/{n} | {o} | {mn} |")

worst = sorted([r for r in rows if r["wer"] > 0.15], key=lambda x: -x["wer"])[:10]
lines.append("")
lines.append("## Худшие файлы (WER>0.15)")
for r in worst:
    lines.append(f"- {r['file']}: wer={r['wer']} judge={r['overall']} mangled={r['mangled']} — {r['issues']}")

open(os.path.join(BASE, "DEV_SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
