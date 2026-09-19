"""Dump pe.py task catalog + demo example groups (tool inventory)."""
import re
import sys

src = open(r"G:\AI\AuK\src\auk\infer\pe.py", encoding="utf-8").read()
m = re.search(r"def _build_classify_prompt\(.*?\n(.*?)\n\n", src, re.S)
if m:
    print("=== CLASSIFY PROMPT (pe.py) ===")
    print(m.group(1)[:1200].encode("ascii", "backslashreplace").decode())
m2 = re.search(r"def _nonverbal_events\(.*?\n(.*?)\n\n\n", src, re.S)
print()
print("=== NONVERBAL EVENTS ===")
if m2:
    print(m2.group(1)[:600].encode("ascii", "backslashreplace").decode())
