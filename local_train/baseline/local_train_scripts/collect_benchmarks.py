"""Collect all Russian-language benchmarks from this project into one report."""
import glob
import json
import os
import sys
from collections import defaultdict

LOCAL = r"G:\AI\AuK\local_train"
LT = r"G:\AI\AuK\local_tests"
OUT = os.path.join(LOCAL, "reports")
os.makedirs(OUT, exist_ok=True)


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def jget(j, k, d=0):
    v = (j or {}).get(k)
    return d if v is None else v


def dir_aggregate(d):
    res = load(os.path.join(d, "results_fixed.json")) or load(os.path.join(d, "results.json"))
    man = load(os.path.join(d, "manifest.json")) or []
    if res is None:
        return None, None, None
    man_map = {m["gen"]: m for m in man}
    rows = []
    for r in res:
        m = man_map.get(r["gen"], {})
        j = r.get("judge") or {}
        recall = r.get("recall_fixed", r.get("recall"))
        rows.append({
            "file": os.path.basename(r["gen"]),
            "voice": m.get("voice") or m.get("representation") or m.get("variant") or "?",
            "seed": m.get("seed"),
            "text": (m.get("text") or "")[:60],
            "overall": jget(j, "overall"),
            "nat": jget(j, "naturalness"),
            "prosody": jget(j, "prosody"),
            "stress": jget(j, "stress"),
            "palat": jget(j, "palatalization"),
            "endings": jget(j, "endings"),
            "recall": recall,
            "missing": r.get("missing_fixed", r.get("missing_words", [])),
            "issues": (j.get("issues") or "")[:120],
        })
    return rows, res, man


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / max(len(vals), 1)


L = []
L.append("# Русскоязычные бенчмарки AuK — сводный отчёт")
L.append("")
L.append("Дата: 2026-09-14. Модель: AuK 1.5B (flow-matching DiT), текстовый энкодер Qwen2.5-Omni-3B (заморожен), VAE (заморожен).")
L.append("LoRA: r32/α64, attention+FF+AdaLayerNorm; fusion/layer-fusion разморожены; bf16; активации-чекпоинтинг.")
L.append("")
L.append("## Методика измерения")
L.append("- **Qwen-судья** (локальный 2.5-Omni-3B, рубрика v2): text_fidelity, endings, naturalness, prosody, accent, palatalization, stress, artifacts, voice_similarity, overall + флаги. Исторически судья склонен к потолку и слеп к части фонетических деталей (см. раздел расхождений).")
L.append("- **ASR-recall/WER**: транскрипция синтеза локальным Qwen-ASR, сравнение с целевым текстом; числа нормализуются, галлюцинации обрезаются, есть ретрай.")
L.append("- **DNSMOS/F0** (watcher): объективные качество/высота голоса на валидационных парах.")
L.append("- **Человеческие оценки** (user_reviews) — приоритетный источник.")
L.append("")

# ---------------- early text representation experiments ----------------
L.append("## 1. Ранние эксперименты: формат текста (до обучения)")
L.append("Короткая фраза, ASR word-recall / качество:")
L.append("")
L.append("| Вариант | Результат (ASR) |")
L.append("|---|---|")
L.append("| Кириллица без разметки | мусор (~10-20% recall) |")
L.append("| Кириллица + комбинир. акут (е́) | нестабильно (0.13-0.39) |")
L.append("| Латиница (транслит) без стресса | хорошо (до 1.00 на короткой фразе) |")
L.append("| Латиница + апостроф-стресс | хорошо (0.85-1.00) |")
L.append("| «ye»-написание (provyerka) | плохо (0.23) |")
L.append("")
exp2 = load(os.path.join(LT, "exp2_transcripts.json"))
exp3 = load(os.path.join(LT, "exp3_transcripts.json"))
if exp2:
    L.append("Длинная фраза (exp2, word-recall):")
    for k, v in sorted(exp2.items()):
        L.append(f"- {k}: recall={v.get('ratio', 0):.2f} | {v.get('transcript','')[:80]}")
    L.append("")
if exp3:
    L.append("Длинная фраза, варианты сидов/написания (exp3, word-recall):")
    for k, v in sorted(exp3.items()):
        L.append(f"- {k}: recall={v.get('recall', 0):.2f} | {v.get('transcript','')[:80]}")
    L.append("")

# ---------------- controller (rubric A) ----------------
L.append("## 2. Контроллер стадии 0/1 (рубрика A, control_rubricA.jsonl)")
rows = []
p = os.path.join(LOCAL, "run_ru_s1", "control_rubricA.jsonl")
if os.path.exists(p):
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        m = r.get("means", {})
        rows.append((r["update"], r["label"], r["score"], m))
    L.append("| update | label | score | overall | accent | palat | nat | sim | wer |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for u, lab, s, m in rows:
        L.append("| u%s | %s | %.3f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f |" % (
            u, lab, s, m.get("overall", 0), m.get("accent", 0), m.get("palatalization", 0),
            m.get("naturalness", 0), m.get("voice_similarity", 0), m.get("wer_mean", 0)))
    L.append("")
    L.append("Победитель стадии 0 по пересуженной рубрике: **u3000** (score 0.786 vs база 0.700).")
    L.append("Лучшая оценка стадии 1 по рубрике A: **u7380 = 0.8778** (overall 8.9, palat 9.6, nat 8.5, sim 8.9).")
    L.append("")

# ---------------- human reviews ----------------
L.append("## 3. Человеческие оценки (user_reviews)")
L.append("| sample | overall | verdict | text/end/nat/pros/stress/palat | issues |")
L.append("|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(LOCAL, "user_reviews", "*.json"))):
    r = load(f)
    if not r:
        continue
    for key in ("verdict_json", "verdict_json_v2"):
        v = r.get(key)
        if not v:
            continue
        L.append("| %s | %s | %s | %s/%s/%s/%s/%s/%s | %s |" % (
            r.get("sample"), v.get("overall"), v.get("verdict"),
            jget(v, "text_fidelity", v.get("text_fidelity")), v.get("endings"), v.get("naturalness"),
            v.get("prosody"), v.get("stress"), v.get("palatalization"), (v.get("issues") or "")[:100]))
L.append("")

# ---------------- variations v2 ----------------
rows2, _, _ = dir_aggregate(os.path.join(LT, "variations_v2"))
if rows2:
    L.append("## 4. Вариативная выборка v2 (42 файла: 20 фраз × 2 сида ± A/B по #14)")
    L.append("MEAN: overall=%.1f nat=%.1f endings=%.1f recall=%.2f (n=%d)" % (
        mean(rows2, "overall"), mean(rows2, "nat"), mean(rows2, "endings"), mean(rows2, "recall"), len(rows2)))
    by_v = defaultdict(list)
    for r in rows2:
        by_v[r["voice"]].append(r)
    for v, sel in sorted(by_v.items()):
        L.append("- %s: overall=%.1f recall=%.2f (n=%d)" % (v, mean(sel, "overall"), mean(sel, "recall"), len(sel)))
    worst = sorted(rows2, key=lambda x: (x["recall"] or 0, -x["overall"]))[:6]
    L.append("")
    L.append("Худшие (по recall):")
    for x in worst:
        L.append("- %s: recall=%.2f missing=%s | %s" % (x["file"], x["recall"] or 0, str(x["missing"])[:50], x["text"]))
    L.append("")

# ---------------- voices pack ----------------
rows3, _, _ = dir_aggregate(os.path.join(LT, "voices_pack"))
if rows3:
    L.append("## 5. Голосовой пак (4 голоса × 4 фразы)")
    by_v = defaultdict(list)
    for r in rows3:
        by_v[r["voice"]].append(r)
    for v, sel in sorted(by_v.items()):
        L.append("- %s: overall=%.1f nat=%.1f (n=%d)" % (v, mean(sel, "overall"), mean(sel, "nat"), len(sel)))
    flagged = [r for r in rows3 if r["overall"] and r["overall"] <= 5]
    if flagged:
        L.append("")
        L.append("Флаги судьи:")
        for r in flagged:
            L.append("- %s (ovrl=%s): %s" % (r["file"], r["overall"], r["issues"]))
    L.append("")
    L.append("Вердикт пользователя: **просодия (роботизированные паузы/сегментация) задевает ВСЕ голосовые сэмплы**, фонетика/окончания/ударения — отличные.")
    L.append("")

# ---------------- stress variants ----------------
rows4, _, _ = dir_aggregate(os.path.join(LT, "stress_variants"))
if rows4:
    L.append("## 6. «Роге» — варианты разметки ударения")
    L.append("| вариант | stress | overall | issues |")
    L.append("|---|---|---|---|")
    for r in rows4:
        L.append("| %s | %s | %s | %s |" % (r["file"], r["stress"], r["overall"], r["issues"][:90]))
    L.append("")
    L.append("Судья и ASR ударение не различают — решает слух пользователя; ждём вердикт.")
    L.append("")

# ---------------- razvitie A/B ----------------
L.append("## 7. A/B слова «развитие» (u7250/u7380)")
L.append("- ref + кириллица **с ударениями**: recall 1.00, «развитие» произнесено верно (оба сида).")
L.append("- ref + кириллица **без ударений**: recall 0.70-0.80, «развитие»/«устойчивое» ломаются («рематы», «реванти»).")
L.append("- ref + транслит (с ' для ь/ъ): в основном ок (1 сбой «играет» на s7).")
L.append("- instruct (без рефа), без ударений: 50/50 (s1234 ок, s7 сломан).")
L.append("")

# ---------------- key divergences ----------------
L.append("## 8. Расхождения судья ↔ человек (важно)")
L.append("- u6500 (человек: **1/10**, «развитие→разитие») — судья: overall 7, endings 9.")
L.append("- var_00_s1234_m (человек: 8/10, микропаузы) — судья: 10/10.")
L.append("- u9250 (человек: пропущен мягкий знак) — судья: palatalization 10.")
L.append("- vol_own_m_p0 (человек: 6/10, роботизированные паузы) — судья: overall 3 ✅ единственное совпадение по слабому месту.")
L.append("→ Численная шкала судьи (обе рубрики) смещена; человеческое ухо — приоритет, судья — грубый фильтр.")
L.append("")

# ---------------- conclusions ----------------
L.append("## 9. Выводы по русскому языку (текущее состояние)")
L.append("**Сильные стороны (стабильно 9-10/10 у судьи и человека):**")
L.append("- Фонетика и акцент: «Standard Russian», без английского акцента (главная цель достигнута на стадии 1).")
L.append("- Окончания слов, палатализация ь/ъ (после кириллического обучения), ударения при наличии разметки RuAccent.")
L.append("- Text fidelity: полный текст без пропусков в подавляющем большинстве (recall ~0.89-1.00).")
L.append("")
L.append("**Слабые стороны (главные задачи):**")
L.append("1. **Просодия/сегментация** — роботизированные паузы на стыках слов, монотонность, «квадратная» редукция. Систематически во всех отзывах. Лечится данными (длинные плавные фразы), возможно параметрами сэмплинга (тест params_ab).")
L.append("2. **Редкие/сложные слова без стресс-разметки** — выпадение слогов («развитие→разитие»); со разметкой RuAccent — исправно. Вывод: держать автоударения включёнными.")
L.append("3. **Числительные** — просодия на числительных роботизирована; ASR их нормализует (метрика учтена).")
L.append("4. **Скороговорки** — теряют слова (var_10, var_18).")
L.append("5. **Instruct-режим (без референса)** слабее ref-режима (клон); UI использует ref-режим — продакшн-качество выше тестовых instruct-сэмплов.")
L.append("")
L.append("**Что дальше (план):**")
L.append("- Стадия 2: смещение данных к длинным фразам ≥4.5 с с пунктуацией; числа прописью+стресс; покрытие сложных слов; replay ~10%.")
L.append("- Протестировать params_ab (NFE/CFG/sway, фактор длительности 0.85/1.15) — уменьшает ли паузы.")
L.append("- Продолжить LR-спад и контроль лучшего чекпоинта (лучший по оценкам: u7380 по судье; по слуху — u8250/u9000/u9500).")
L.append("")

report = os.path.join(OUT, "RUSSIAN_BENCH_REPORT.md")
open(report, "w", encoding="utf-8").write("\n".join(L))
print("report written:", report, "|", len(L), "lines")
for x in L[1:8]:
    print(x)
