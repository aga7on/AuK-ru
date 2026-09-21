# -*- coding: utf-8 -*-
"""Проверка целостности human_gate shortlist: wav-ссылки разрешаются, тексты не пустые."""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else r"G:\AI\AuK\local_tests\human_gate_s16c"
rows = json.load(open(os.path.join(d, "shortlist.json"), encoding="utf-8"))
miss = sum(1 for r in rows for k in ("a", "b")
           if not os.path.exists(os.path.normpath(os.path.join(d, r[k]))))
empty = sum(1 for r in rows if not r.get("text"))
html = [f for f in os.listdir(d) if f.startswith("listen_")][0]
t = open(os.path.join(d, html), encoding="utf-8").read()
print(f"{os.path.basename(d)}: pairs={len(rows)} missing_wav={miss} empty_text={empty} html={html}")
print("flags A+B:", "жуёт слова (A)" in t and "жуёт слова (B)" in t,
      "| both-bad:", "оба плохи" in t, "| LS_KEY unique:", t.count("LS_KEY='auk_human_gate"))
