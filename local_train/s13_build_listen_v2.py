# -*- coding: utf-8 -*-
"""Пересборка listen.html (S13): добавлены чекбоксы «жуёт слова» для A и B,
опция «оба плохи», экспорт флагов в CSV. Данные пар встраиваются из LISTEN.csv.
"""
import csv
import io
import json

PAIR_DIR = r"G:\AI\AuK\local_tests\pairwise_v1"

rows = list(csv.DictReader(open(PAIR_DIR + r"\LISTEN.csv", encoding="utf-8")))
data = [{"id": r["id"], "kind": r["kind"], "text": r["text"],
         "a": r["a_wav"], "b": r["b_wav"]} for r in rows]
print("pairs:", len(data))

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>AuK-ru pairwise A/B — слепое прослушивание v2</title>
<style>
 body{font-family:system-ui,Arial;background:#111;color:#ddd;margin:24px;max-width:1020px}
 h1{font-size:20px} .hint{color:#999;font-size:13px;margin-bottom:18px;line-height:1.5}
 .pair{border:1px solid #333;border-radius:8px;padding:12px 14px;margin:12px 0;background:#181818}
 .pid{font-weight:bold;color:#8ab4f8} .kind{color:#aaa;font-size:12px;margin-left:8px}
 .text{margin:6px 0 10px;color:#eee}
 .row{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .side{display:flex;flex-direction:column;gap:6px}
 audio{height:34px}
 .flags label{font-size:13px;color:#f0b429;cursor:pointer;user-select:none}
 .btns{margin-left:auto;display:flex;gap:8px;align-items:center}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:7px 14px;cursor:pointer;font-size:14px}
 button.sel-A{background:#1d4ed8;border-color:#3b82f6;color:#fff}
 button.sel-B{background:#b91c1c;border-color:#ef4444;color:#fff}
 button.sel-X{background:#555;border-color:#888;color:#fff}
 .done{color:#4ade80;font-size:12px}
 #bar{position:sticky;top:0;background:#111;padding:10px 0;border-bottom:1px solid #333;z-index:9}
 #export{background:#166534;border-color:#22c55e}
 .stats{color:#8ab4f8;font-size:13px;margin-left:12px}
</style>
</head>
<body>
<h1>Слепое A/B v2: 40 пар + флаги дикции</h1>
<div class="hint">
Для каждой пары: прослушайте A и B, отметьте галочкой <b>«жуёт слова»</b> у тех вариантов, где слова
проглатываются/произносятся нечётко (можно у обоих, можно ни у одного). Затем выберите лучший:
<b>A</b>, <b>B</b> или <b>оба плохи</b> (если оба с браком и выбирать нечего).
Прогресс и флаги сохраняются в браузере автоматически. В конце — «Экспорт CSV» и пришлите файл агенту.
</div>
<div id="bar">
  <span id="progress"></span>
  <span class="stats" id="flagstats"></span>
  &nbsp; <button id="export" onclick="exportCsv()">Экспорт CSV</button>
  <button onclick="clearAll()">Сброс</button>
</div>
<div id="list"></div>
<script>
const LS_KEY = 'auk_pairwise_v2';
let pairs = [];
let state = JSON.parse(localStorage.getItem(LS_KEY) || '{}');
// state[id] = {choice: 'A'|'B'|'X', flagA: bool, flagB: bool}

const EMBEDDED = __DATA__;
pairs = EMBEDDED;
render();

function render() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  for (const p of pairs) {
    const st = state[p.id] || {};
    const div = document.createElement('div');
    div.className = 'pair';
    div.innerHTML = `
      <div><span class="pid">${p.id}</span><span class="kind">${p.kind}</span></div>
      <div class="text">«${p.text}»</div>
      <div class="row">
        <div class="side">
          <div>A: <audio controls preload="none" src="${p.a}"></audio></div>
          <div class="flags"><label><input type="checkbox" id="fA_${p.id}" ${st.flagA ? 'checked' : ''}
            onchange="setFlag('${p.id}','flagA',this.checked)"> жуёт слова (A)</label></div>
        </div>
        <div class="side">
          <div>B: <audio controls preload="none" src="${p.b}"></audio></div>
          <div class="flags"><label><input type="checkbox" id="fB_${p.id}" ${st.flagB ? 'checked' : ''}
            onchange="setFlag('${p.id}','flagB',this.checked)"> жуёт слова (B)</label></div>
        </div>
        <div class="btns">
          <button id="btnA_${p.id}" onclick="pick('${p.id}','A')">A лучше</button>
          <button id="btnB_${p.id}" onclick="pick('${p.id}','B')">B лучше</button>
          <button id="btnX_${p.id}" onclick="pick('${p.id}','X')">оба плохи</button>
          <span class="done" id="ok_${p.id}"></span>
        </div>
      </div>`;
    list.appendChild(div);
    if (st.choice) markChosen(p.id, st.choice);
  }
  updateProgress();
}

function getState(id){ if(!state[id]) state[id]={}; return state[id]; }
function save(){ localStorage.setItem(LS_KEY, JSON.stringify(state)); updateProgress(); }

function pick(id, side) {
  getState(id).choice = side;
  save(); markChosen(id, side);
}

function setFlag(id, key, val) {
  getState(id)[key] = val;
  save();
}

function markChosen(id, side) {
  for (const s of ('A','B','X')) {
    const b = document.getElementById(`btn${s}_${id}`);
    if (b) b.className = (s === side) ? ('sel-' + s) : '';
  }
  const ok = document.getElementById('ok_' + id);
  if (ok) ok.textContent = side === 'X' ? '✓ оба плохи' : ('✓ ' + side);
}

function updateProgress() {
  const answered = Object.values(state).filter(s => s && s.choice).length;
  const fA = Object.values(state).filter(s => s && s.flagA).length;
  const fB = Object.values(state).filter(s => s && s.flagB).length;
  document.getElementById('progress').textContent = `Выбрано: ${answered} / ${pairs.length}`;
  document.getElementById('flagstats').textContent = `Флагов «жуёт»: A=${fA}, B=${fB}`;
}

function exportCsv() {
  let csv = 'id,kind,choice,flag_a_mumbled,flag_b_mumbled\\n';
  for (const p of pairs) {
    const st = state[p.id] || {};
    csv += `${p.id},${p.kind},${st.choice || ''},${st.flagA ? 1 : 0},${st.flagB ? 1 : 0}\\n`;
  }
  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pairwise_v2_choices.csv';
  a.click();
}

function clearAll() {
  if (!confirm('Сбросить все выборы и флаги?')) return;
  state = {};
  localStorage.removeItem(LS_KEY);
  render();
}
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
io.open(PAIR_DIR + r"\listen.html", "w", encoding="utf-8", newline="\n").write(html)
print("listen.html v2 written, bytes:", len(html.encode("utf-8")))
