# -*- coding: utf-8 -*-
"""Разведка апстрима Tencent-Hunyuan/AuK: ветки, коммиты, Flash-код, файлы релиза.

Цель (запрос пользователя 09.11): узнать о новых ветках развития и о том, выпустили ли
разработчики инструмент для обучения Flash-модели (AuK-Flash: 4-step, Decoupled DMD).

Пишет отчёт в local_train/reports/deepseek_supervised/UPSTREAM_WATCH.md
Ничего не меняет в репозитории.

usage: python probe_upstream.py
"""
import json
import os
import subprocess
import sys

REPO = "Tencent-Hunyuan/AuK"
OUT = r"G:\AI\AuK\local_train\reports\deepseek_supervised\UPSTREAM_WATCH.md"

BRANCHES = ["main", "dev-ziyang", "dev-wenming", "feat/PE", "feat/local-pe",
            "feat/mlx-apple-silicon"]


def api(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def main():
    lines = ["# UPSTREAM WATCH — Tencent-Hunyuan/AuK", "",
             "Собрано автоматически `probe_upstream.py`. Источник: GitHub API + страница проекта.",
             ""]

    info = api(f"repos/{REPO}")
    if info:
        lines += ["## Репозиторий", "",
                  f"- default_branch: `{info.get('default_branch')}`",
                  f"- pushed_at: {info.get('pushed_at')}",
                  f"- size: {info.get('size')} KB",
                  f"- stars: {info.get('stargazers_count')}",
                  f"- license: {((info.get('license') or {}).get('spdx_id'))}",
                  ""]

    lines += ["## Ветки", "", "| ветка | последний коммит | дата | ahead/behind main |", "|---|---|---|---|"]
    branch_data = {}
    for b in BRANCHES:
        br = api(f"repos/{REPO}/branches/{b}")
        if not br:
            lines.append(f"| {b} | (нет доступа) | — | — |")
            continue
        sha = br.get("commit", {}).get("sha")
        msg = ((br.get("commit", {}).get("commit") or {}).get("message") or "").split("\n")[0]
        date = ((br.get("commit", {}).get("commit") or {}).get("author") or {}).get("date", "")
        cmp_ = api(f"repos/{REPO}/compare/main...{b}") if b != "main" else None
        ab = "—"
        if cmp_:
            ab = f"+{cmp_.get('ahead_by', 0)} / -{cmp_.get('behind_by', 0)}"
        branch_data[b] = {"sha": sha, "msg": msg, "date": date}
        lines.append(f"| `{b}` | {msg[:70]} | {date[:10]} | {ab} |")
    lines.append("")

    # поиск Flash / DMD / distillation в коде каждой ветки
    KEYS = ("flash", "dmd", "distill", "consistency", "decoupled", "4-step", "four_step")
    lines += ["## Признаки Flash / дистилляции (поиск по путям файлов)", "",
              "| ветка | файлы с ключевыми словами |", "|---|---|"]
    flash_found = {}
    for b in BRANCHES:
        tree = api(f"repos/{REPO}/git/trees/{b}?recursive=1")
        if not tree:
            lines.append(f"| `{b}` | (нет доступа) |")
            continue
        hits = []
        for t in tree.get("tree", []):
            p = t.get("path", "")
            if any(k in p.lower() for k in KEYS):
                hits.append(p)
        flash_found[b] = hits
        lines.append(f"| `{b}` | {len(hits)} |")
        for h in hits[:12]:
            lines.append(f"| | `{h}` |")
    lines.append("")

    # верхнеуровневые каталоги каждой ветки — понять структуру (есть ли train для flash)
    lines += ["## Верхний уровень (структура)", ""]
    for b in BRANCHES:
        tree = api(f"repos/{REPO}/git/trees/{b}")
        if not tree:
            continue
        tops = sorted({t["path"] for t in tree.get("tree", [])})
        lines.append(f"- `{b}`: {', '.join(tops[:20])}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("branches:")
    for b, d in branch_data.items():
        print(f"  {b:26s} {d['date'][:10]} {d['msg'][:60]}")
    print("flash/dmd path hits per branch:")
    for b, h in flash_found.items():
        print(f"  {b:26s} {len(h)}")
        for x in h[:6]:
            print("      ", x)
    print("wrote", OUT)


if __name__ == "__main__":
    main()