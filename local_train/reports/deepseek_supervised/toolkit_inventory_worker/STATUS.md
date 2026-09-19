# STATUS — toolkit_inventory_worker

Phase: corrective review + root-authorized 9-WAV diagnostic synthesis.
Last update: 2026-09-16 22:27 (run finished).

## 1. Correction phase (done)

Rewrote `INVENTORY.md`, `EXECUTION_PLAN.md`, and corrected
`eval_manifest_proposal.json` (via `correct_manifest.py`):

- full sha256 from `../full_sha256.json` used for controls; the first-2GiB values
  are now labelled `sha256_first_2GiB_partial` (not "full").
- heldout wording: `path_absent_from_finetune_manifests` / `dev_monitored_val` /
  `untouched_final` / `unknown_vs_upstream_pretrain`; no pretraining-exclusion claim.
- clone/tool pack refs reclassified as **legitimate dev/monitored-val** (not gradient
  contamination); tools speaker-cluster overlap 317 separated per `../TOOLS_OVERLAP.md`.
- PE catalog vs raw model: 6 dB / 0.9x / 1.1x / 1.2x are **off-template
  generalization**, not invalid.
- effect eligibility (`deaccent`/`enhancement`/`quality`) = **unverified** unless
  acoustic evidence; `vocal_extraction` speech-only = modality-invalid.
- implicit nonverbal anchor not declared invalid.
- removed the `gen_seconds provided` invalid rule (source-length gen_seconds is valid).
- instruct-TTS fixture corrected (zh instruction + English text; 5.0 s), plus new
  English no-ref short fixture; no Chinese-output claim.
- metrics: Gemini judge (root-delegated) + objective corroboration; human optional.
  Corrected DSP `dsp_review_worker/dsp_core_v2.py` recommended; `tool_dsp_measure.py`
  superseded; raw waveform corr explicitly not voice similarity / content correctness.
- provenance claim restricted to tracked files verified via `git diff`/`git show`
  (only 3 modified files; `pe.py`/`pe.config.yaml` byte-identical to HEAD).

## 2. Diagnostic synthesis (done)

- Pack: `diag_pack.json` (sha256 `4fa4cc32…`), 3 fixtures / 3 controls, 0 errors.
- Controls full sha256 verified against `../full_sha256.json` (`diag_models_sha.json`):
  `upstream_base 29c65c0c…`, `s1_u10000 a9ac0b81…`, `s2_B500 2cef557e…` — all match.
- GPU check: GPU0 7.2 GiB used / 84-86% util (competing workload, **not killed**);
  GPU1 1.3 GiB / idle → runner auto-selected **cuda:1**.
- Runner: `run_diag.py`, public `AukInfer` only, explicit config/ckpt paths, one engine
  at a time, engine released between controls. seed 1234, nfe 64, cfg 2.0, trim false,
  bestofn 1, bf16, cpu_offload. **No clipping normalization, no silence trim.**
- Measured runtime: **136.0 s wall** for 9/9 items (engine load 18.5 / 13.0 / 11.9 s;
  per-item 7.1–14.7 s). Errors 0, no_result 0. 9 unique output paths + 9 unique sha256.
- No Gemini/judge calls this phase (root delegates judging).

## 3. Exact output manifest

| control | fixture | status | sha256 | dur s | peak | clip>=.985 | s |
|---|---|---|---|---|---|---:|---:|---:|---:|
| upstream_base | diag_pitch_en_up2 | ok | ea7a90549ad2eb330d47846c95ed76714d7002cc74397bd5fc2bdd4fc3e987ce | 5.5 | 0.342 | 0.0 | 14.7 |
| upstream_base | diag_zeroshot_en | ok | a9bb36113b0955da5bf33e83e630a15a74d02be8e65a3f8cecc535014cf44679 | 7.0 | 0.473 | 0.0 | 12.5 |
| upstream_base | diag_instructtts_en_noref | ok | d4f55b504927380a380859fe83e88900d07b082a84ec6fe6c7fbe65261095332 | 5.0 | 0.680 | 0.0 | 8.1 |
| s1_u10000 | diag_pitch_en_up2 | ok | ea73b4c8db044be9777ca2461eb05cf0fc3e4906084fb7439fc2843eecae43ca | 5.5 | 0.346 | 0.0 | 8.9 |
| s1_u10000 | diag_zeroshot_en | ok | f09c63160cdb3ef44e357cca5fd09258625bf02c9bbecd260065115835d5cf03 | 7.0 | 0.448 | 0.0 | 10.8 |
| s1_u10000 | diag_instructtts_en_noref | ok | a1ea1299c38298c3dacfb120ad6a8b881d455f26220ee49c8024c5c199a32e10 | 5.0 | 0.676 | 0.0 | 7.1 |
| s2_B500 | diag_pitch_en_up2 | ok | f58c121e9695dbffd376bc5b5ce7effc2265669996b3cf31dd5aada42ce1aed1 | 5.5 | 0.366 | 0.0 | 9.4 |
| s2_B500 | diag_zeroshot_en | ok | f645134369bb003c16c284b053de4eb3cfe34ad4d589484fd6e0f86d15ff6019 | 7.0 | 0.415 | 0.0 | 12.1 |
| s2_B500 | diag_instructtts_en_noref | ok | 0b43b343536e5f4c7558b174244b2e930fa37bc46535750cca2ea3766d30f2b3 | 5.0 | 0.798 | 0.0 | 8.2 |

Outputs: `diag_out/<control>/<fixture>.wav`. Manifest: `diag_out/progress_manifest.json`
(atomic, per-item). Stdout: `diag_out/run_stdout.log`. Model hashes: `diag_models_sha.json`.

## 4. Process state

No lingering diagnostic process; GPU1 memory returned to baseline (~1.34 GiB).
Pre-existing unrelated processes (FoxMCP, watch_samples) untouched.

## 5. Not done / next (root)

- Judging of these 9 with correct modalities (Gemini) — root-delegated, not done here.
- No expansion to the full 27-fixture set.
- No further synthesis or LoRA training.
