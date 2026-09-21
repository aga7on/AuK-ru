# -*- coding: utf-8 -*-
"""Поиск provenance обучающих данных: HF-репозитории, лицензии, метаданные загрузки.

Зачем: релиз должен указывать авторство/лицензии использованных датасетов и инструментов.
README датасетов — автогенерированные HF-карточки без поля license, поэтому ищем:
  1. *.metadata в .cache/huggingface/download (там лежит исходный ETag/commit HF);
  2. имена каталогов → известные HF-наборы (Golos=SberDevices, Common Voice=Mozilla,
     FLEURS=Google, Sova=sova.ai, OpenSTT=Snakers4);
  3. любые файлы с упоминанием license/CC BY/CDLA.

usage: python probe_data_provenance.py
"""
import json
import os
import re
import sys

ROOT = r"G:\AI\kyutai-ru\data"


def main():
    metas = []
    for dp, dn, fn in os.walk(ROOT):
        # не лезем вглубь больших медиа-каталогов
        dn[:] = [d for d in dn if d not in ("latents", "ru_wav", "ru_wav_mfa", "iashchak",
                                            "golos_mfa", "golos_balalaika", "openstt_balalaika")]
        for f in fn:
            if f.endswith(".metadata") or f in ("dataset_infos.json", "README.md",
                                                "LICENSE", "LICENSE.md"):
                metas.append(os.path.join(dp, f))
        if len(metas) > 400:
            break
    print("metadata/license candidates:", len(metas))

    lic_hits = []
    for p in metas:
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        m = re.search(r"(license\w*|cc[- ]by|cdla|apache|mit|non-?commercial|cc[- ]0)[^\n]{0,120}",
                      t, re.I)
        if m and "License-Expression" not in m.group(0):
            lic_hits.append((p, m.group(0)[:150]))
    print("\n--- license mentions ---")
    for p, s in lic_hits[:25]:
        print(os.path.relpath(p, ROOT)[:70], "::", s[:120])
    if not lic_hits:
        print("(none found in dataset dirs)")

    print("\n--- .cache metadata contents (source HF refs) ---")
    n = 0
    for p in metas:
        if not p.endswith(".metadata"):
            continue
        try:
            t = open(p, encoding="utf-8", errors="replace").read().strip()
        except Exception:
            continue
        if not t:
            continue
        print(os.path.relpath(p, ROOT)[:70], "->", t[:110].replace("\n", " "))
        n += 1
        if n >= 12:
            break
    if n == 0:
        print("(metadata files empty or absent)")

    print("\n--- dirs and sizes ---")
    for d in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        tot, cnt = 0, 0
        for dp, dn, fn in os.walk(full):
            for f in fn:
                try:
                    tot += os.path.getsize(os.path.join(dp, f))
                    cnt += 1
                except OSError:
                    pass
            if cnt > 20000:
                break
        print(f"{d:24s} files~{cnt:6d} size~{tot/1e9:6.2f} GB")


if __name__ == "__main__":
    main()
