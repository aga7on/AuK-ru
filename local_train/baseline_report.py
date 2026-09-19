"""Базовое качество русского на u10000: первые попытки, ошибки текста, голос.

Вход: pack.json + каталог прогона (wav + опц. manifest.json).
Метрики: GigaAM ASR -> WER/CER (ru_metrics), флаги аудио, косинус WeSpeaker vs ref,
опц. судейский слой (bridge, --judge). Первичный критерий попытки: wer<=0.15 + нет
ASR_ERROR/пустого/клиппинга (для capability — только запись, вердикт вручную).

usage: baseline_report.py --pack <pack.json> --out <u0_dir> --report-dir <dir> [--judge]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

AUK = r"G:\AI\AuK"
sys.path.insert(0, os.path.join(AUK, "src"))
sys.path.insert(0, os.path.join(AUK, "local_train"))

MANIFEST = r"G:\AI\kyutai-ru\data\ru_mfa_manifest.jsonl"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_manifest_texts(ids):
    want = set(ids)
    found = {}
    if not want:
        return found
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            cid = os.path.splitext(os.path.basename(o["path"]))[0]
            if cid in want:
                found[cid] = o["transcript"]
                if len(found) == len(want):
                    break
    return found


def expected_text(item, manifest_texts):
    kind = item["kind"]
    if kind == "tts":
        return item.get("text") or ""
    if kind == "clone":
        m = re.search(r"'([^']*)'", item["instruction"])
        return m.group(1) if m else ""
    if kind == "tool":
        m = re.search(r"([0-9a-f]{16})", os.path.basename(item["ref"]))
        cid = m.group(1) if m else None
        return manifest_texts.get(cid, f"<no manifest for {cid}>")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--votes", type=int, default=2)
    ap.add_argument("--judge-model", default="gemini-3.8-flash-medium")
    args = ap.parse_args()

    from ru_metrics import transcribe_path, text_metrics, audio_metrics, asr_flags
    from speaker_embed import embed_path

    pack = json.load(open(args.pack, encoding="utf-8"))
    os.makedirs(args.report_dir, exist_ok=True)

    tool_ids = set()
    for it in pack:
        if it["kind"] == "tool":
            m = re.search(r"([0-9a-f]{16})", os.path.basename(it["ref"]))
            if m:
                tool_ids.add(m.group(1))
    manifest_texts = load_manifest_texts(tool_ids)
    print(f"manifest: {len(manifest_texts)}/{len(tool_ids)} tool source transcripts", flush=True)

    emb_cache = {}

    def emb(p):
        if p not in emb_cache:
            emb_cache[p] = embed_path(p)
        return emb_cache[p]

    asr_cache = {}

    def asr(p):
        if p not in asr_cache:
            asr_cache[p] = transcribe_path(p)
        return asr_cache[p]

    rows = []
    judge_subset = []
    for i, it in enumerate(pack):
        wav = os.path.join(args.out, f"{it['id']}.wav")
        if not os.path.exists(wav):
            rows.append({"id": it["id"], "kind": it["kind"], "status": "missing"})
            continue
        exp = expected_text(it, manifest_texts)
        heard, asr_err = transcribe_path(wav)
        tm = text_metrics(exp, heard) if exp else None
        am = audio_metrics(wav)
        flags = asr_flags(heard)
        if asr_err:
            flags.append("ASR_ERROR")
        ref = it.get("ref")
        sim = None
        ref_heard = None
        if ref and os.path.exists(ref):
            try:
                sim = cosine(emb(wav), emb(ref))
            except Exception as e:
                flags.append(f"EMB_ERR:{type(e).__name__}")
            ref_heard, _ = asr(ref)
        leak = None
        if tm and ref_heard:
            leak_m = text_metrics(ref_heard, heard)
            if tm["wer"] > 0.3 and leak_m["wer"] < tm["wer"] - 0.15:
                leak = True
                flags.append("REF_LEAK?")
        row = {
            "id": it["id"], "kind": it["kind"], "op": it.get("op") or it.get("cap"),
            "expected": exp, "heard": heard, "asr_error": asr_err,
            "wer": tm["wer"] if tm else None, "cer": tm["cer"] if tm else None,
            "ref_heard": ref_heard, "leak": leak,
            "sim": round(sim, 4) if sim is not None else None,
            "flags": flags,
            "dur": am["duration"], "clip_ratio": am["clip_ratio"], "empty": am["empty"],
            "peak": am["peak"], "n_pauses": am["n_pauses"], "pause_max": am["pause_max"],
        }
        if tm:
            row["subs"] = tm.get("subs", [])
            row["dels"] = tm.get("dels", [])
            row["inss"] = tm.get("inss", [])
        text_ok = bool(tm and tm["wer"] <= 0.15 and not asr_err and not am["empty"])
        row["first_ok"] = text_ok if it["kind"] != "capability" else None
        row["first_ok_raw"] = bool(text_ok and am["clip_ratio"] <= 0.0002) if it["kind"] != "capability" else None
        rows.append(row)

        caps = {"tts": 12, "clone": 6, "tool": 6, "capability": 12}
        idx = {"tts": 4, "clone": 4, "tool": 6, "capability": 1}[it["kind"]]
        if args.judge and i % idx == 0 and sum(1 for x in judge_subset if x["kind"] == it["kind"]) < caps[it["kind"]]:
            judge_subset.append({"id": it["id"], "kind": it["kind"], "wav": wav, "text": exp})

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(pack)}]", flush=True)

    if args.judge and judge_subset:
        from ru_metrics import judge_audio

        print(f"judge: {len(judge_subset)} items x votes={args.votes}", flush=True)
        by_id = {r["id"]: r for r in rows}
        for k, js in enumerate(judge_subset):
            rubric, jerr = judge_audio(js["wav"], js["text"] or " ", votes=args.votes,
                                       model=args.judge_model)
            by_id[js["id"]]["judge"] = rubric
            by_id[js["id"]]["judge_error"] = jerr
            print(f"  [{k+1}/{len(judge_subset)}] {js['id']} "
                  f"{'no_result' if rubric is None else json.dumps(rubric, ensure_ascii=False)[:80]}", flush=True)

    ok_rows = [r for r in rows if r.get("status") != "missing"]
    agg = {}
    for kind in ("tts", "clone", "tool", "capability"):
        rr = [r for r in ok_rows if r["kind"] == kind]
        text_r = [r for r in rr if r.get("wer") is not None]
        oks = [r["first_ok"] for r in rr if r.get("first_ok") is not None]
        wers = [r["wer"] for r in text_r]
        sims = [r["sim"] for r in rr if r.get("sim") is not None]
        judge_vals = [r["judge"].get("overall") or r["judge"].get("score") for r in rr
                      if isinstance(r.get("judge"), dict) and (r["judge"].get("overall") or r["judge"].get("score"))]
        agg[kind] = {
            "n": len(rr),
            "first_ok": (round(sum(oks) / len(oks), 3) if oks else None),
            "first_ok_raw": (round(sum(1 for r in rr if r.get("first_ok_raw")) / len(oks), 3) if oks else None),
            "wer_mean": round(float(np.mean(wers)), 3) if wers else None,
            "wer_median": round(float(np.median(wers)), 3) if wers else None,
            "sim_median": round(float(np.median(sims)), 3) if sims else None,
            "judge_mean": round(float(np.mean(judge_vals)), 2) if judge_vals else None,
            "flag_counter": dict(Counter(f.split(":")[0] for r in rr for f in r.get("flags", []))),
        }

    tts_by_voice = {}
    for r in ok_rows:
        if r["kind"] != "tts":
            continue
        voice = r["id"].rsplit("_", 2)[-2] + "_" + r["id"].rsplit("_", 1)[-1]
        tts_by_voice.setdefault(voice, []).append(r)
    voice_agg = {v: {"n": len(rr), "sim_median": round(float(np.median([r["sim"] for r in rr if r.get("sim")])), 3)
                     if any(r.get("sim") for r in rr) else None,
                     "wer_mean": round(float(np.mean([r["wer"] for r in rr if r.get("wer") is not None])), 3)
                     if any(r.get("wer") is not None for r in rr) else None}
                 for v, rr in tts_by_voice.items()}

    worst = sorted([r for r in ok_rows if r.get("wer") is not None], key=lambda r: -r["wer"])[:12]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": args.out, "pack": args.pack, "judge_used": bool(args.judge),
        "aggregate": agg, "tts_by_voice": voice_agg,
        "items": rows,
    }
    json.dump(report, open(os.path.join(args.report_dir, "baseline_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    L = []
    L.append("# Базовое качество u10000 (u0-контроль, первые попытки)")
    L.append("")
    L.append(f"- Дата: {report['generated_at']}; каталог: `{args.out}`")
    L.append("- Протокол: без best-of-N, без трима; NFE=64, seed=1234. Судейский слой — доп. сигнал, не ground truth.")
    L.append("- Критерий удачной первой попытки: WER ≤ 0.15, нет ASR_ERROR/пустого (для capability — вручную).")
    L.append("")
    L.append("## Итоги по категориям")
    L.append("")
    L.append("| Категория | n | удачных первых (текст) | + без клиппинга | WER mean | WER median | sim median | judge mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for kind, a in agg.items():
        L.append(f"| {kind} | {a['n']} | {a['first_ok']} | {a['first_ok_raw']} | {a['wer_mean']} | "
                 f"{a['wer_median']} | {a['sim_median']} | {a['judge_mean']} |")
    L.append("")
    if voice_agg:
        L.append("## TTS по голосам (sim vs ref)")
        L.append("")
        L.append("| Голос | n | sim median | WER mean |")
        L.append("|---|---|---|---|")
        for v, a in sorted(voice_agg.items()):
            L.append(f"| {v} | {a['n']} | {a['sim_median']} | {a['wer_mean']} |")
        L.append("")
    L.append("## Худшие 12 по WER")
    L.append("")
    for r in worst:
        L.append(f"- **{r['id']}** WER={r['wer']:.2f} sim={r['sim']} flags={r['flags']}")
        L.append(f"  - expected: {r['expected'][:90]}")
        L.append(f"  - heard:    {r['heard'][:90]}")
    L.append("")
    caps = [r for r in ok_rows if r["kind"] == "capability"]
    if caps:
        L.append("## Capability (прослушивание вручную)")
        L.append("")
        for r in caps:
            j = r.get("judge")
            L.append(f"- **{r['id']}** sim={r['sim']} dur={r['dur']} flags={r['flags']}"
                     + (f" judge={json.dumps(j, ensure_ascii=False)[:100]}" if j else ""))
            L.append(f"  - heard: {r['heard'][:110]}")
    L.append("")
    L.append("## Оговорки")
    L.append("- ASR слеп к ударению/мягкости части фонем — окончательный вердикт за прослушиванием.")
    L.append("- Клиппинг: u0-прогон без limit_peak; продуктовая цепочка его сглаживает.")
    L.append("- Инструменты: корректность операции проверяется отдельно (аудио-эффекты), здесь — сохранение содержания.")
    md = "\n".join(L)
    open(os.path.join(args.report_dir, "BASELINE_U0.md"), "w", encoding="utf-8").write(md)
    print(md[:3000])
    print(f"\nsaved: {args.report_dir}\\baseline_report.json + BASELINE_U0.md")


if __name__ == "__main__":
    main()
