# -*- coding: utf-8 -*-
"""S16 human-gate builder: слепой набор для ПРОСЛУШИВАНИЯ hard_eval_v1_clean (235 позиций).

Дизайн (по решению пользователя 20.09):
  automatic screening (ASR + оси судьи + mumble-прокси)
    → candidate shortlist (подозрительные + случайная выборка чистых)
    → human blind test (listen html, A vs B: v1.0 vs s16, порядок рандомизирован)
    → PASS/FAIL по метрикам: human_mumble_rate, both_bad_rate, win-rate.

Этап 1 (этот скрипт, --shortlist): screening двух прогонов frontend_probe
(v1.0 = frontend_probe_s9_v4, s16 = frontend_probe_s16) → shortlist до 24 пар.
Этап 2 (--build): сборка listen_s16.html + SECRET-карты (только после генераций).

usage:
  python s16_human_gate.py --shortlist          # после обоих проб
  python s16_human_gate.py --build --src_a <dirA> --src_b <dirB>
"""
import argparse
import csv
import io
import json
import os
import random
import sys

sys.path.insert(0, r"G:\AI\AuK\src")
sys.path.insert(0, r"G:\AI\AuK\local_train")

AUK = r"G:\AI\AuK"
D = os.path.join(AUK, "local_train", "reports", "deepseek_supervised")
OUT_DIR = os.path.join(AUK, "local_tests", "human_gate_s16")
SEED = 161


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def screening(probe_dir, judge_jsonl):
    """Возвращает {id: {wer, heard, judge_axes...}} для одной модели."""
    res = {r["id"]: r for r in json.load(open(os.path.join(probe_dir, "results.json"), encoding="utf-8"))
           if r.get("status") == "ok"}
    judged = {}
    for r in jl(judge_jsonl):
        tid = r.get("task_id", "")
        if r.get("status") == "ok":
            base = tid.split("__")[0]
            judged[base] = r["judge"]
    out = {}
    for i, r in res.items():
        j = judged.get(i, {})
        # probe-результаты хранят исходный текст в "raw", нормализованный — в "expected"
        # (поля "text" там нет) — берём raw для показа человеку, expected как запасной
        text = r.get("raw") or r.get("expected") or r.get("text") or ""
        out[i] = {"file": r["file"], "text": text, "expected": r.get("expected"),
                  "category": r.get("category"),
                  "wer": r.get("wer_norm", r.get("wer")), "cer": r.get("cer_norm"),
                  "fid": j.get("text_fidelity"), "accent": j.get("accent"),
                  "palat": j.get("palatalization"), "stress": j.get("stress"),
                  "nat": j.get("naturalness"), "verdict": j.get("verdict"),
                  "mangled": j.get("words_mangled") or [], "subs": j.get("phoneme_substitutions") or []}
    return out


def shortlist(a, b, n_pairs=24):
    """Приоритет: (1) расхождения A/B по CER/WER/осям; (2) подозрительные у обоих;
    (3) случайные чистые (контроль unbiased). CER — первичная метрика (правило ROADMAP 3)."""
    ids = sorted(set(a) & set(b))
    scored = []
    for i in ids:
        ra, rb = a[i], b[i]
        s = 0
        # CER — основной (весо 1.0), WER — вспомогательный (0.5, искажается сегментацией ASR)
        for k, w in (("cer", 1.0), ("wer", 0.5)):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                s += abs(va - vb) * w
        for k in ("fid", "nat"):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                s += abs(va - vb) * 0.3
        susp = ((ra.get("cer") or 0) > 0.15 or (rb.get("cer") or 0) > 0.15
                or (ra.get("wer") or 0) > 0.5 or (rb.get("wer") or 0) > 0.5)
        if susp:
            s += 1
        scored.append((s, i))
    scored.sort(reverse=True)
    n_top = (n_pairs * 2) // 3
    top = [i for _, i in scored[:n_top]]
    rest = [i for _, i in scored[n_top:]]
    rng = random.Random(SEED)
    ctrl = rng.sample(rest, min(n_pairs - len(top), len(rest)))
    return top + ctrl


def build_html(pairs_rows, title):
    rows_js = json.dumps(pairs_rows, ensure_ascii=False)
    return """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:system-ui,Arial;background:#111;color:#ddd;margin:24px;max-width:1020px}
 h1{font-size:19px} .hint{color:#999;font-size:13px;margin-bottom:16px;line-height:1.5}
 .pair{border:1px solid #333;border-radius:8px;padding:12px 14px;margin:12px 0;background:#181818}
 .pid{font-weight:bold;color:#8ab4f8} .kind{color:#aaa;font-size:12px;margin-left:8px}
 .text{margin:6px 0 10px;color:#eee}
 .row{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .side{display:flex;flex-direction:column;gap:6px}
 audio{height:34px}
 .flags label{font-size:13px;color:#f0b429;cursor:pointer}
 .btns{margin-left:auto;display:flex;gap:8px;align-items:center}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:7px 14px;cursor:pointer;font-size:14px}
 button.sel-A{background:#1d4ed8;border-color:#3b82f6;color:#fff}
 button.sel-B{background:#b91c1c;border-color:#ef4444;color:#fff}
 button.sel-X{background:#555;border-color:#888;color:#fff}
 .done{color:#4ade80;font-size:12px}
 #bar{position:sticky;top:0;background:#111;padding:10px 0;border-bottom:1px solid #333;z-index:9}
 #export{background:#166534;border-color:#22c55e}
</style></head><body>
<h1>__TITLE__</h1>
<div class="hint">Слепое сравнение двух версий модели на ТРУДНЫХ текстах (frozen hard_eval_v1).
Отметьте «жуёт слова» у каждого варианта, где слышите проглатывание/нечёткость; затем выберите
лучший: A / B / оба плохи. Порядок рандомизирован. Экспорт CSV в конце — пришлите файл агенту.</div>
<div id="bar"><span id="progress"></span> <span id="flagstats" style="color:#8ab4f8;font-size:13px;margin-left:12px"></span>
 &nbsp; <button id="export" onclick="exportCsv()">Экспорт CSV</button> <button onclick="clearAll()">Сброс</button></div>
<div id="list"></div>
<script>
const LS_KEY='auk_human_gate_s16';
let pairs=__ROWS__;
let state=JSON.parse(localStorage.getItem(LS_KEY)||'{}');
render();
function render(){const list=document.getElementById('list');list.innerHTML='';
 for(const p of pairs){const st=state[p.id]||{};const div=document.createElement('div');div.className='pair';
  div.innerHTML=`<div><span class="pid">${p.id}</span><span class="kind">${p.category||''}</span></div>
   <div class="text">«${p.text}»</div>
   <div class="row">
    <div class="side"><div>A: <audio controls preload="none" src="${p.a}"></audio></div>
     <div class="flags"><label><input type="checkbox" ${st.flagA?'checked':''} onchange="setFlag('${p.id}','flagA',this.checked)"> жуёт слова (A)</label></div></div>
    <div class="side"><div>B: <audio controls preload="none" src="${p.b}"></audio></div>
     <div class="flags"><label><input type="checkbox" ${st.flagB?'checked':''} onchange="setFlag('${p.id}','flagB',this.checked)"> жуёт слова (B)</label></div></div>
    <div class="btns"><button id="btnA_${p.id}" onclick="pick('${p.id}','A')">A лучше</button>
     <button id="btnB_${p.id}" onclick="pick('${p.id}','B')">B лучше</button>
     <button id="btnX_${p.id}" onclick="pick('${p.id}','X')">оба плохи</button>
     <span class="done" id="ok_${p.id}"></span></div></div>`;
  list.appendChild(div); if(st.choice) markChosen(p.id, st.choice);}
 updateProgress();}
function gs(id){if(!state[id])state[id]={};return state[id];}
function save(){localStorage.setItem(LS_KEY,JSON.stringify(state));updateProgress();}
function pick(id,s){gs(id).choice=s;save();markChosen(id,s);}
function setFlag(id,k,v){gs(id)[k]=v;save();}
function markChosen(id,s){for(const x of('A','B','X')){const b=document.getElementById(`btn${x}_${id}`);if(b)b.className=(x===s)?('sel-'+x):'';}
 const ok=document.getElementById('ok_'+id);if(ok)ok.textContent=s==='X'?'✓ оба плохи':'✓ '+s;}
function updateProgress(){const n=Object.values(state).filter(s=>s&&s.choice).length;
 document.getElementById('progress').textContent=`Выбрано: ${n} / ${pairs.length}`;
 const fa=Object.values(state).filter(s=>s&&s.flagA).length, fb=Object.values(state).filter(s=>s&&s.flagB).length;
 document.getElementById('flagstats').textContent=`Жуёт: A=${fa}, B=${fb}`;}
function exportCsv(){let csv='id,category,choice,flag_a_mumbled,flag_b_mumbled\\n';
 for(const p of pairs){const st=state[p.id]||{};
  csv+=`${p.id},${p.category||''},${st.choice||''},${st.flagA?1:0},${st.flagB?1:0}\\n`;}
 const blob=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='human_gate_s16_choices.csv';a.click();}
function clearAll(){if(!confirm('Сбросить всё?'))return;state={};localStorage.removeItem(LS_KEY);render();}
</script></body></html>""".replace("__TITLE__", title).replace("__ROWS__", rows_js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--src_a", default=os.path.join(AUK, "local_tests", "frontend_probe_s9_v4"))
    ap.add_argument("--src_b", default=os.path.join(AUK, "local_tests", "frontend_probe_s16"))
    ap.add_argument("--judge_a", default=os.path.join(D, "s16_baseline_phonetics_results.jsonl"))
    ap.add_argument("--judge_b", default=os.path.join(D, "s16_phonetics_results.jsonl"))
    ap.add_argument("--n_pairs", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.shortlist:
        a = screening(args.src_a, args.judge_a)
        b = screening(args.src_b, args.judge_b)
        ids = shortlist(a, b, args.n_pairs)
        rows = []
        rng = random.Random(SEED)
        secret = []
        for i in ids:
            swap = rng.random() < 0.5
            va, vb = ("v1", "s16") if not swap else ("s16", "v1")
            fa = a[i]["file"] if not swap else b[i]["file"]
            fb = b[i]["file"] if not swap else a[i]["file"]
            rows.append({"id": i, "category": a[i].get("category"), "text": (a[i].get("text") or "")[:160],
                         "a": os.path.relpath(fa, OUT_DIR).replace("\\", "/"),
                         "b": os.path.relpath(fb, OUT_DIR).replace("\\", "/")})
            secret.append({"id": i, "a_variant": va, "b_variant": vb})
        json.dump(rows, open(os.path.join(OUT_DIR, "shortlist.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        with open(os.path.join(OUT_DIR, "SECRET_variant_map.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "a_variant", "b_variant"])
            w.writeheader()
            w.writerows(secret)
        print("shortlist:", len(rows), "pairs ->", OUT_DIR)
    if args.build:
        rows = json.load(open(os.path.join(OUT_DIR, "shortlist.json"), encoding="utf-8"))
        html = build_html(rows, "S16 human gate — hard_eval_v1 (v1.0 vs s16, слепо)")
        io.open(os.path.join(OUT_DIR, "listen_s16.html"), "w", encoding="utf-8", newline="\n").write(html)
        print("listen_s16.html built:", len(rows), "pairs")


if __name__ == "__main__":
    main()
