# EXECUTION_PLAN — heldout evaluation of original AuK capabilities

Worker: `toolkit_inventory_worker`. This is a **plan only**: no synthesis, training,
or judge calls were run. Proposed new output dirs live outside this worker's owned
folder and therefore require root authorization to create.

Companion files (this folder): `INVENTORY.md`, `eval_manifest_proposal.json`,
`assets_probe.json`, `leakage_report.json`, `probe_assets.py`, `check_leakage.py`.
Controls: `upstream_base` = `ckpts/AuK/auk_base.safetensors`,
`s1_u10000` = `local_train/run_ru_s1/merged/auk_ru_10000.safetensors`.
Shared protocol: seed 1234, nfe 64, cfg 2.0, trim false (matches
`local_train/run_eval_pack.py` and the existing 5-variant packs).

---

## 1. Existing commands that already run

```powershell
# Read-only provenance/leakage (this worker, already executed):
.venv\Scripts\python.exe local_train\reports\deepseek_supervised\toolkit_inventory_worker\probe_assets.py
.venv\Scripts\python.exe local_train\reports\deepseek_supervised\toolkit_inventory_worker\check_leakage.py

# Objective DSP over an existing 5-variant tool pack (already executed by the other worker):
.venv\Scripts\python.exe local_train\tool_dsp_measure.py

# Existing eval-pack runner (flat list: id/kind/instruction/ref/gen_seconds/text):
.venv\Scripts\python.exe local_train\run_eval_pack.py --pack local_tests\eval_pack\pack.json `
  --ckpt ckpts\AuK\auk_base.safetensors --out local_tests\toolkit_eval\upstream_base --device cuda:0
```

The manifest runner works unchanged for any **flat** pack. `run_eval_pack.py`
reads `config.yaml` next to the checkpoint, auto-passes `ckpts/Qwen2.5-Omni-3B`,
uses `cpu_offload=True`, and writes `manifest.json`; it does **not** write
`run_params.json` (add it, see §3).

---

## 2. Runtime / storage assumptions (approximate, must be confirmed)

Assumptions: 1× CUDA GPU, bf16, `cpu_offload=True`, nfe=64, cfg=2.0, clips
3–30 s, per-fixture generation ≈ 10–20 s including ASR scoring (cross-checked
against the existing run timestamps: each 120-item variant run ≈ 18 min,
`STATUS.md:31-32`).

| Block | items | runs | est. generation | est. storage (24 kHz mono PCM16) |
|---|---:|---:|---:|---:|
| P0 smoke | 3 | 2 (×2 controls) | < 2 min | < 5 MB |
| P1 original native control (19 fixtures) | 19 | 38 | 15–30 min | ~300 MB |
| P2 Russian phonetics (existing) | 16 | 5 (already done) | reuse | ~80 MB |
| P3 re-fixtured instrumented tools (proposed) | 7 ops × 5 = 35 | 2 controls (×5 variants if A/B needed) | 15–60 min | ~150–400 MB |
| P4 synthetic media sets (proposed) | ~15 | 2 controls | 20–40 min | ~300 MB |
| Final frozen test (proposed) | 6 | 2 controls | 5–10 min | ~100 MB |
| Checkpoints on disk (not copied) | — | — | — | 6.1 GB ×2 + Qwen 3B + VAE |

Disk headroom needed if all blocks run: **≈ 1.5 GB** of wavs, negligible vs the
≈ 18 GB of checkpoints/VAE already present. Load time per cold engine start ≈ 15 s
(`infer_gradio.py:473`).

---

## 3. Precise script requirements (new files, needs authorization)

### 3.1 Adapter: manifest → flat pack
Requirement: read `eval_manifest_proposal.json`, keep fixtures with
`status == "valid"`, emit a flat JSON list with keys
`id, kind, instruction, ref, gen_seconds, text`. For `equal_length` tasks an
explicit `gen_seconds` equal to the actual source duration is **valid and
preferred** for cross-model determinism; otherwise pass `None` (do not coerce to 0
— `infer_auk.py:287-291` then uses source length). No other logic.

### 3.2 `run_params.json` for reproducibility
Requirement: after each run, write (mirroring existing dirs): `ckpt`,
`ckpt_sha16` (first 2 GiB, `local_train/verify_eval_run.py::sha16`), `pack`,
`pack_sha16`, `pack_entries`, `seed`, `nfe`, `cfg_strength`, `trim`, `bestofn`,
`model_alias`, `prompt_version`, `status`, `errors`, `missing`. Full sha256 of
merged files exists in `local_train/reports/deepseek_supervised/full_sha256.json`.

### 3.3 Metrics (corrected)
- **Perceptual judge**: Gemini audio judge via the local proxy — root delegates the
  judging; human listening is **optional, not a blocker**. Per-fixture criteria must
  name the judge, not "2 listeners".
- **Objective magnitudes**: use the **sibling corrected** DSP
  `dsp_review_worker/dsp_core_v2.py` (see `dsp_v2_controls.json`). The older
  `tool_dsp_measure.py` is **superseded** — do not use its SNR / preservation
  correlation as a pass metric.
- **Never** use raw waveform correlation as voice similarity or content correctness.
- Speaker similarity: `local_train/speaker_embed.py` (ONNX wespeaker cosine).
- Russian content: `src/auk/infer/quality.py::wer_metrics` (NOT `recall`).
- zh/en content: no local ASR; Gemini judge + optional human.

### 3.4 Synthetic media builders (P4, needs explicit authorization)
- `build_separation_mixtures.py`: sum two held-out single-speaker recordings at
  fixed gains/offsets; emit mix + two stems + transcripts. Source eligibility and
  forbidden inputs are in `eval_manifest_proposal.json → media_gaps[0]`.
- `build_quality_effects.py`: apply fixed deterministic degradations
  (megaphone/underwater/clipping/dropout/dc_offset) to an upstream clean source;
  record exact parameters. See `media_gaps[2]`.
- Music separation: do **not** synthesize from speech-only audio
  (`media_gaps[1]`); either reuse `assets/demo-input-audio/vocal-extraction/vocal-1-input.wav`
  or wait for independently licensed singing+instrumental stems.

### 3.5 Authorized 9-WAV diagnostic (executed now; see `STATUS.md`)
Three fixtures × three controls (`upstream_base`, `s1_u10000`, `s2_B500`):
`diag_pitch_en_up2`, `diag_zeroshot_en`, `diag_instructtts_en_noref`.
Independent runner + pack live in this folder (`build_diag_pack.py`,
`run_diag.py`, `diag_pack.json`); outputs under
`toolkit_inventory_worker/diag_out/<control>/`. No Gemini calls in this phase.


---

## 4. Prioritized smallest-intervention set

Ship the smallest set that can falsify "original capability preserved":

### P0 — harness smoke (do first, cheapest)
3 fixtures × 2 controls: `smoke_upstream_pitch_up2`, `smoke_ru_pitch_up2_heldout_ref`,
`orig_zero_shot_tts_en`. Pass = DSP control still valid (`TOOL_DSP.md` controls)
+ English TTS intelligible. Confirms both checkpoints load and the runner works.

### P1 — full original native control (the core deliverable)
All 19 `status=valid` original fixtures (§`eval_manifest_proposal.json`),
**both controls**, zh/en task language. This is the only block that actually
measures whether Russian fine-tuning preserved upstream capability. Paired
criterion on every fixture; absolute criteria once thresholded on the **dev** split.

### P2 — Russian speech + phonetics (existing assets, no new build)
Reuse `local_tests/phonetic_pack/pack.json` (16) and the `phon_*` variant dirs;
re-score with `wer_metrics` (never `recall`) and keep the blind form. This is the
"speech phonetics" leg of the requested smallest set.

### P3 — tools with valid, in-catalog, heldout fixtures
Replace the current tool block: (a) only documented operands (5/10/15 dB,
0.5/0.75/1.25/1.5/2.0×, 1/2/3 st); (b) sources held out from
`local_train/data*`, `data_s2_full*`, `data_s2_tools_v3*`; (c) objective DSP
metric per op. Until rebuilt, treat existing `tool_*` results as
non-heldout/non-catalog (`INVENTORY.md:2.1.1,2.1.8`).

### P4 — media-gap expansions (separation, quality effects, real accented ru)
Build only after P1 passes; see `media_gaps` and §3.4.

---

## 5. Test/dev vs final split (protocol)

- **Dev split (tunable):** all 19 original fixtures + `phonetic_pack`. Used to set
  numeric thresholds and to debug the DSP/speaker metrics. May be re-run.
- **Final split (frozen, single reveal):** pick a disjoint-by-purpose subset that
  is *never* used for tuning and is scored once with recorded output sha256:
  1. `orig_zero_shot_tts_en`
  2. `orig_content_edit_replace_en`
  3. `orig_vocal_edit_lyrics_en`
  4. `orig_music_separation_singing_only`
  5. `orig_target_speaker_extraction_en`
  6. `orig_instruct_tts_en_noref_short`
  Perceptual scoring by the Gemini judge (root-delegated), with objective
  corroboration; human listening optional. Freeze the pack file sha256 before
  generation; open no result-derived tuning afterwards.
- **Speaker/recording isolation:** verify final fixtures with the
  `check_leakage.py` method; classify each as train / monitored val-dev /
  untouched final. The current `clone/*` and `tool/*` refs are **dev/monitored-val**
  (legitimate development eval, not gradient contamination; tool val has 317
  speaker-cluster overlaps with s2 speech train per `TOOLS_OVERLAP.md`), so they
  must not be the *final* set.
- **No `SECRET_map.csv` opening, no `listen_form.csv` edits, no judge calls** in
  this plan's default path.

---

## 6. Explicit non-goals / prohibitions honored

No training, no synthesis, no Gemini calls, no background jobs, no secrets
printed, no shared docs/AGENTS/evaluator modified, no checkpoints deleted. All
new files are inside `local_train/reports/deepseek_supervised/toolkit_inventory_worker/`;
the proposed output dirs in §1–§4 are declared for authorization, not created.
