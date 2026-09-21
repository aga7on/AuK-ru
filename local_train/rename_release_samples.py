# -*- coding: utf-8 -*-
"""Переименование релизных семплов в честные имена + подготовка фактов.

Причина: исходные имена (01_female_happy и т.д.) ложные — ASR+f0 показали, что все 5
файлов мужским голосом, 01/03/05 — один текст, эмоции акустически неразличимы
(EMOTION_SEPARABILITY.md: η²<0.10 у 6/7, LOO 0.017≈chance, Cohen d≈0).

Новые имена: sample_01..05_male.wav (пол — фактический, без ложных emotion/женских меток).
usage: python rename_release_samples.py [--src tmp_samples_check/samples] [--dst release/samples]
"""
import argparse
import json
import os
import shutil

AUK = r"G:\AI\AuK"
RENAME = {
    "01_female_happy.wav": "sample_01_male.wav",
    "02_female_excited.wav": "sample_02_male.wav",
    "03_male_sad.wav": "sample_03_male.wav",
    "04_male_angry.wav": "sample_04_male.wav",
    "05_male_fearful.wav": "sample_05_male.wav",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(AUK, "local_train", "tmp_samples_check", "samples"))
    ap.add_argument("--dst", default=os.path.join(AUK, "local_train", "release", "samples"))
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for old, new in RENAME.items():
        s = os.path.join(args.src, old)
        d = os.path.join(args.dst, new)
        if os.path.exists(s):
            shutil.copyfile(s, d)
            print(f"{old} -> {new} ({os.path.getsize(d)} B)")
        else:
            print("MISSING src:", old)
    print("wrote to", args.dst)


if __name__ == "__main__":
    main()
