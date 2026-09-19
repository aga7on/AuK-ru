"""П.4: ревью-пак отклонённых клипов — 150-200 штук с причинами и heard-текстом.
Проверяющий слушает аудио и смотрит: исходный транскрипт vs GigaAM-текст.
Причины: вставки / замены / числа / окончания / высокий WER / GigaAM-fail.
"""
import json
import os
import random
import sys
from collections import defaultdict

LOCAL = r"G:\AI\AuK\local_train"
CORPUS = os.path.join(LOCAL, "corpus")
OUT = os.path.join(LOCAL, "tests_packs", "transcript_review")
sys.path.insert(0, LOCAL)
sys.path.insert(0, r"G:\AI\AuK\src")


def classify_reason(m):
    if m["recall"] >= 0.9 and m["insertions"]:
        return "insertions"
    if m["recall"] < 0.5:
        return "high_wer"
    if any("#" in s[0] or "#" in s[1] for s in m["substitutions"]):
        return "numbers"
    # смотреть, если замена на 1-2 буквы в конце → окончания
    end_subs = [s for s in m["substitutions"] if s[0][:3] == s[1][:3]]
    if len(end_subs) == len(m["substitutions"]) and len(m["substitutions"]) <= 2 and len(m["deletions"]) <= 1:
        return "endings"
    if m["deletions"] and not m["insertions"] and len(m["deletions"]) == 1:
        return "sound_drop"
    if m["wer"] <= 0.2:
        return "minor_wer"
    return "substitutions"


def main():
    import soundfile as sf
    import shutil

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(11)
    from ru_metrics import text_metrics, transcribe_path

    import shutil
    manifest = []
    targets = {"insertions": 15, "numbers": 15, "high_wer": 15, "substitutions": 40,
               "minor_wer": 40, "endings": 15, "sound_drop": 15}
    got = defaultdict(int)
    MAX_TOTAL = 160

    # пары отклонены из-за identity — добавить только если не хватает
    pairs_pool = []
    with open(os.path.join(CORPUS, "s2_pairs_v2.jsonl"), encoding="utf-8") as f:
        for line in f:
            pairs_pool.append(json.loads(line))
    rng.shuffle(pairs_pool)

    with open(os.path.join(CORPUS, "split_manifest.jsonl"), encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    rng.shuffle(lines)

    for o in lines[:6000]:
        if sum(got.values()) >= MAX_TOTAL:
            break
        if not os.path.exists(o["p"]):
            continue
        heard, err = (None, None)
        from ru_metrics import transcribe_path, text_metrics
        heard, err = transcribe_path(o["p"])
        if err:
            continue
        m = text_metrics(o.get("t", ""), heard)
        reason = classify_reason(m)
        if got[reason] >= targets.get(reason, 10):
            continue
        fname = f"rv{len(manifest):03d}_{reason}.wav"
        dst = os.path.join(OUT, fname)
        shutil.copy(o["p"], dst)
        manifest.append({
            "file": fname, "reason": reason, "audio": dst,
            "original": o.get("t", "")[:200], "gigaam_heard": heard[:200],
            "wer": m["wer"], "recall": m["recall"], "insertions": m["insertions"],
            "deletions": m["deletions"], "substitutions": m["substitutions"][:5],
            "duration": o.get("d"),
            "question": "Сравни транскрипты: какая версия верная? [оригинал/GigaAM/оба не точны/не решено]",
        })
        got[reason] += 1
        if len(manifest) % 40 == 0:
            print(f"  {len(manifest)} collected ({dict(got)})", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\nREVIEW_PACK_DONE: {len(manifest)} items -> {OUT}")
    print("by reason:", dict(got))


if __name__ == "__main__":
    main()
