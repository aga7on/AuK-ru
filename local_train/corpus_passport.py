"""Шаг 4: паспорт корпуса. Проход по манифестам, нормализация путей, распределения.

Выход: local_train/corpus/CORPUS_PASSPORT.md + corpus_stats.json
"""
import json
import os
import re
from collections import Counter, defaultdict

D = r"G:\AI\kyutai-ru\data"
OUT = r"G:\AI\AuK\local_train\corpus"

MANIFESTS = [
    ("ru_train_aligned.jsonl", "train_aligned"),
    ("ru_valid_aligned.jsonl", "valid_aligned"),
    ("ru_all.jsonl", "all"),
    ("ru_mfa_manifest.jsonl", "mfa_manifest"),
]

_PATH_RE = re.compile(r'"path": "([^"]+)"')
_DUR_RE = re.compile(r'"duration": ([0-9.]+)')
_TRANS_RE = re.compile(r'"transcript": "([^"]{0,400})"')


def norm_path(p: str) -> str:
    p = p.replace("/", "\\")
    if p.startswith("X:\\MediaForge\\kyutai-ru-data\\"):
        return "G:\\AI\\kyutai-ru\\data\\" + p[len("X:\\MediaForge\\kyutai-ru-data\\"):]
    return p


def guess_source(path: str) -> str:
    low = path.lower()
    if "common_voice" in low:
        return "common_voice"
    if "golos" in low:
        return "golos"
    if "openstt" in low or "balalaika" in low:
        return "openstt_balalaika"
    if "sber" in low:
        return "sber"
    if "sova" in low or "fleurs" in low:
        return "sova_fleurs"
    if "iashchak" in low:
        return "iashchak"
    return "unknown"


def bucket(d):
    if d < 2:
        return "0-2s"
    if d < 4:
        return "2-4s"
    if d < 6:
        return "4-6s"
    if d < 8:
        return "6-8s"
    if d < 10:
        return "8-10s"
    if d < 15:
        return "10-15s"
    return "15s+"


def scan(path, kind, full_stat=False, sample_stat=3000):
    n = 0
    dur_sum = 0.0
    buckets = Counter()
    sources = Counter()
    short = 0
    long_ = 0
    wps_vals = []
    exists_ok = 0
    exists_checked = 0
    checked_paths = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            n += 1
            pm = _PATH_RE.search(line)
            dm = _DUR_RE.search(line)
            tm = _TRANS_RE.search(line)
            p = norm_path(pm.group(1)) if pm else ""
            d = float(dm.group(1)) if dm else 0.0
            dur_sum += d
            buckets[bucket(d)] += 1
            sources[guess_source(p)] += 1
            if d < 2:
                short += 1
            if d > 15:
                long_ += 1
            if tm and d > 0.5:
                nwords = len(tm.group(1).split())
                wps_vals.append(nwords / d)
            if (full_stat or (i % max(1, (n and 1) or 1) == 0)) and p:
                if full_stat or i % max(1, 400) == 0:
                    if len(checked_paths) < sample_stat or full_stat:
                        checked_paths.append(p)
    # existence check (sampled unless full_stat)
    for p in checked_paths:
        exists_checked += 1
        if os.path.exists(p):
            exists_ok += 1
    stats = {
        "manifest": os.path.basename(path),
        "kind": kind,
        "clips": n,
        "hours": round(dur_sum / 3600, 2),
        "duration_buckets": dict(buckets),
        "sources": dict(sources),
        "clips_under_2s": short,
        "clips_over_15s": long_,
        "words_per_sec_mean": round(sum(wps_vals) / max(len(wps_vals), 1), 2),
        "path_exists_sampled": f"{exists_ok}/{exists_checked}",
    }
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    all_stats = []
    for fname, kind in MANIFESTS:
        p = os.path.join(D, fname)
        if not os.path.exists(p):
            all_stats.append({"manifest": fname, "missing": True})
            continue
        full = kind in ("train_aligned", "valid_aligned")
        st = scan(p, kind, full_stat=full)
        all_stats.append(st)
        print(f"{fname}: {st.get('clips')} clips, {st.get('hours')}h, exists {st.get('path_exists_sampled')}", flush=True)

    lines = ["# CORPUS PASSPORT (шаг 4)", "", f"Источник: {D}", ""]
    for st in all_stats:
        if st.get("missing"):
            lines.append(f"## {st['manifest']} — НЕ НАЙДЕН")
            continue
        lines.append(f"## {st['manifest']} ({st['kind']})")
        lines.append(f"- клипов: **{st['clips']}**, часы: **{st['hours']}**")
        lines.append(f"- файлы на месте (выборка): {st['path_exists_sampled']}")
        lines.append(f"- источники: {json.dumps(st['sources'], ensure_ascii=False)}")
        lines.append(f"- длительности: {json.dumps(st['duration_buckets'], ensure_ascii=False)}")
        lines.append(f"- <2с: {st['clips_under_2s']}, >15с: {st['clips_over_15s']}, слов/сек: {st['words_per_sec_mean']}")
        lines.append("")
    lines.append("## Спикеры")
    lines.append("В манифестах НЕТ speaker_id → шаг 5 (восстановление) обязателен. "
                 "Восстановление: по именам файлов резервов (golos/balalaika/sber хранят записи говорящих) + кластеризация.")
    lines.append("")
    lines.append("## Выводы v0")
    lines.append("- Основной массив: ru_wav (87k файлов, 18.4 GB) + ru_wav_mfa (942k, 168.7 GB).")
    lines.append("- aligned-манифесты дают word-timestamps (MFA) — годятся для пар 'реф≠цель' и оценки границ.")
    lines.append("- Разделение train/dev/test возможно только после шага 5 (спикеры/записи).")
    open(os.path.join(OUT, "CORPUS_PASSPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(all_stats, open(os.path.join(OUT, "corpus_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("PASSPORT_DONE")


if __name__ == "__main__":
    main()
