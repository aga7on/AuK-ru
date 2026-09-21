# -*- coding: utf-8 -*-
"""Описание релизных семплов: что реально сказано (ASR) + объективный пол/подача.

Зачем: имена файлов в релизе обещали female/male + эмоцию. Пользователь прослушал и
сообщил, что голос мужской и эмоций нет. Нужны факты, чтобы подписать семплы честно.

usage: python describe_samples.py [--dir local_train/tmp_samples_check/samples]
Выход: stdout (таблица) + local_train/tmp_samples_check/samples_facts.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"G:\AI\AuK\local_train\tmp_samples_check\samples")
    ap.add_argument("--out", default=r"G:\AI\AuK\local_train\tmp_samples_check\samples_facts.json")
    args = ap.parse_args()

    from ru_metrics import transcribe_path

    rows = []
    for fn in sorted(os.listdir(args.dir)):
        if not fn.endswith(".wav"):
            continue
        p = os.path.join(args.dir, fn)
        heard, err = transcribe_path(p)
        rows.append({"file": fn, "heard": heard, "asr_error": err,
                     "size": os.path.getsize(p)})
        print(f"{fn}: {heard!r} (err={err})", flush=True)

    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
