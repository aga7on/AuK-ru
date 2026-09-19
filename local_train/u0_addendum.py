"""Дополнение к BASELINE_U0.md: разбор утечек по голосам + интерпретация."""
import json
from collections import defaultdict

r = json.load(open(r"G:\AI\AuK\local_train\reports\u0_baseline\baseline_report.json", encoding="utf-8"))
items = [x for x in r["items"] if x.get("status") != "missing"]

byv = defaultdict(lambda: {"n": 0, "leak": 0, "ok": 0})
for it in items:
    if it["kind"] != "tts":
        continue
    v = "_".join(it["id"].split("_")[-2:])
    d = byv[v]
    d["n"] += 1
    d["leak"] += 1 if "REF_LEAK?" in it["flags"] else 0
    d["ok"] += 1 if it.get("first_ok") else 0

clone = [x for x in items if x["kind"] == "clone"]
tool = [x for x in items if x["kind"] == "tool"]
caps = [x for x in items if x["kind"] == "capability"]
cap_sims = sorted(x["sim"] for x in caps if x["sim"])
cap_med = cap_sims[len(cap_sims) // 2] if cap_sims else None

L = []
L.append("")
L.append("## Дополнение: механика базовой картины (16.09)")
L.append("")
L.append("Рефы: user_ref 10.21s | ref_ru 8.74s | ref_male2 6.06s | ref_female2 7.50s.")
L.append("Все ≤ 10.2s — реф-кап продукта не влияет; утечка = сырое поведение модели.")
L.append("")
L.append("| TTS голос | n | ok | утечка рефа |")
L.append("|---|---|---|---|")
for v, d in sorted(byv.items()):
    L.append(f"| {v} | {d['n']} | {d['ok']} | {d['leak']} |")
L.append("")
L.append(f"- Клонирование: утечек {sum(1 for x in clone if 'REF_LEAK?' in x['flags'])}/{len(clone)}, "
         f"ok {sum(1 for x in clone if x.get('first_ok'))}.")
L.append(f"- Инструменты: утечек {sum(1 for x in tool if 'REF_LEAK?' in x['flags'])}/{len(tool)}, "
         f"ok {sum(1 for x in tool if x.get('first_ok'))}.")
L.append(f"- Capability: sim median {round(cap_med, 3) if cap_med else None} — требуется прослушивание.")
L.append("")
L.append("Ключевой механизм брака в СЫРОМ режиме: на коротком целевом тексте модель иногда")
L.append("«продолжает» реф (произносит его слова) — раньше закрывалось продуктовыми ретраями")
L.append("(base → t3+tight → tight → template3) и best-of-N по WER. В этом прогоне ретраев и")
L.append("выбора НЕТ (протокол A/B). alt_male/alt_female страдают сильнее — это и есть предмет")
L.append("пилота: снижает ли s2 долю таких сырых срывов и не портит ли tools-примесь речь.")
L.append("")

with open(r"G:\AI\AuK\local_train\reports\u0_baseline\BASELINE_U0.md", "a", encoding="utf-8") as f:
    f.write("\n".join(L))
print("\n".join(L))
