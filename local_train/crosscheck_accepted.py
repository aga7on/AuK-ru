"""П.5: перекрёстная проверка принятых — судья + GigaAM."""
import json
import os
import re
import sys

sys.path.insert(0, r"G:\AI\AuK\local_train")
sys.path.insert(0, r"G:\AI\AuK\src")
from ru_metrics import text_metrics, transcribe_path

man = json.load(open(r"G:\AI\AuK\local_tests\s2_pack_check\manifest.json", encoding="utf-8"))
res = json.load(open(r"G:\AI\AuK\local_tests\s2_pack_check\results_gemini.json", encoding="utf-8"))
jres = {x["file_name"]: (x.get("judge") or {}) for x in res}

print("=== ПЕРЕКРЁСТНАЯ ПРОВЕРКА: судья + GigaAM на принятых ===")
for m in man:
    fn = os.path.basename(m["file"])
    j = jres.get(fn, {})
    text = m["text"]
    match = re.search(r"'([^']+)'", text)
    clean = match.group(1) if match else text
    heard, err = transcribe_path(m["file"])
    tm = text_metrics(clean, heard)
    o = j.get("overall", "?")
    mangled = j.get("words_mangled", [])
    flag = "!" if tm["wer"] > 0.15 else " "
    print(f"{flag}{fn}: judge={o} wer={tm['wer']:.2f} recall={tm['recall']:.2f} ins={len(tm['insertions'])} mangled={mangled}")
    if tm["wer"] > 0.15:
        print(f"   heard: {heard[:80]}")
