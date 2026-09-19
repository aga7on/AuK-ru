"""Проверка поддержки ref-аудио в тренировочном пайплайне AuK."""
import os
import re

SRC = r"G:\AI\AuK\src\auk\train"
for root, dirs, files in os.walk(SRC):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        src = open(p, encoding="utf-8").read()
        for pat in ("ref_audio", '"audio"', "'audio'", "no_prompt_audio"):
            for m in re.finditer(re.escape(pat), src):
                line_no = src[: m.start()].count("\n") + 1
                line = src.splitlines()[line_no - 1].strip()[:110]
                print(f"{f}:{line_no} [{pat}] {line.encode('ascii','backslashreplace').decode()}")
