# INVENTORY — Original AuK capability surface and heldout-evaluation coverage

Worker: bounded DeepSeek (`toolkit_inventory_worker`). Date: 2026-09-16.
Method: read-only inspection of the local upstream checkout
(`origin = https://github.com/Tencent-Hunyuan/AuK.git`, `HEAD = 0dfd4d39015078351217b09a40e263355aaa646b`,
2026-09-13) plus the local Russian adaptation artifacts. Raw evidence:
`assets_probe.json`, `leakage_report.json` (this folder). Sibling corrected DSP:
`local_train/reports/deepseek_supervised/dsp_review_worker/`.

> Scope rule: only capabilities written in upstream source/docs or exercised by
> upstream assets are counted. Nothing is asserted as "supported" from the Russian
> fine-tune. Uncertainty is marked `unknown`.

---

## 0. Provenance separation (corrected)

| Layer | Location | Exact evidence |
|---|---|---|
| Upstream tracked, tracked-and-unmodified | `README.md`, `docs/`, `src/auk/infer/pe.py`, `src/auk/infer/pe.config.yaml`, `assets/demo-input-audio/**`, `scripts/train.sh` | `git diff --name-only HEAD` lists only 3 files, **none of them these**; `git diff --quiet HEAD -- src/auk/infer/pe.py` exit 0; same for `pe.config.yaml` (byte-identical to HEAD) |
| Local uncommitted edits (NOT upstream) | `src/auk/infer/infer_auk.py` (+1 line bf16 cast), `src/auk/infer/infer_gradio.py`, `src/auk/train/train.py` | `git diff --name-only HEAD` = these 3 only |
| Untracked additions (NOT upstream) | `src/auk/infer/{gigaam_ctc,quality,ru_translit}.py`, `local_tests/`, `local_train/`, `assets/after_pe/` (gitignored) | `git status --porcelain` |

`git status`/`git diff` compare the working tree to HEAD for **tracked** files only;
untracked additions can coexist with a clean diff, so "pristine" is claimed **only**
for the specific tracked files above (verified by diff), not for the whole subtree.

### Model controls (full sha256 from `../full_sha256.json`)

| Control | Path | Full sha256 | Size |
|---|---|---|---|
| `upstream_base` | `ckpts/AuK/auk_base.safetensors` | `29c65c0c6045e8d8fb454019f99c9680508f553fc98fa143feaca3711d0b8614` | 6 122 209 092 |
| `s1_u10000` | `local_train/run_ru_s1/merged/auk_ru_10000.safetensors` | `a9ac0b81f5159ce707afc31e4897be2236716163144447092f7a0e0908642a24` | 6 122 209 092 |
| `s2_B500` | `local_train/run_s2_B/merged/auk_s2_B_500.safetensors` | `2cef557ee9be95c5bb5748c51265b7e57ce3f613923bb1c3177e4f9de8c8daf2` | 6 122 209 092 |

Partial-hash note (correction): `assets_probe.json` also stores a **sha256 over the
first 2 GiB only** (`upstream_base` `cf5a4ec2…`, `s1_u10000` `4f3eca35…`). These are
**prefix hashes, not full-file hashes**, and are retained only to cross-check the
historical `ckpt_sha16` convention (which hashes the first 2048 MB). Use the full
hashes above for identity.

---

## 1. Original task categories, subtypes, modalities, duration semantics

Sources: `README.md:82-182`, `docs/COOKBOOK.md`, `src/auk/infer/pe.config.yaml`
(unmodified upstream), `src/auk/infer/infer_auk.py`, `src/auk/infer/infer_cli.py`.

Two distinct layers matter and must not be conflated:

- **Raw model contract** (`infer_auk.py` / `infer_cli.py`): one ChatML user turn; a
  text `instruction` is always required; an optional `{type:audio, audio|audio_url}`
  item supplies the reference/source; with no audio the runtime appends
  `|<no_prompt_audio>|` (`infer_auk.py:279-284`). Audio is downmixed to mono and
  resampled to 24 kHz (`docs/FINETUNING.md:53`). The raw model will attempt **any**
  instruction/operand; there is no enum validation at inference.
- **PE catalog** (`pe.config.yaml`): enumerates the discrete values PE may emit
  (volume 5/10/15 dB; speed 0.5/0.75/1.25/1.5/2.0×; pitch 1/2/3 st; 8 emotions; …).
  This constrains **PE's output**, not the raw model's capability. A raw-pipeline
  test at 6 dB or 1.1× is an **off-template generalization probe**, not an invalid
  task.

### 1.1 Capability table

| # | README category | PE `task_type` | Subtypes / discrete values (PE catalog) | Input modality | Duration semantics |
|---|---|---|---|---|---|
| 1 | Zero-shot TTS | `zero_shot_tts` | — (`text`) | ref audio + text | `gen_seconds` > (`ref_text`+`gen_text` ratio) > source length (`infer_auk.py:287-291,341-358`; `infer_cli.py:33`) |
| 2 | Instruct TTS | `instruct_tts` | — (`style_desc`+`text`) | text only | raw CLI requires a length (`infer_cli.py:64-66`); PE estimates (`pe.config.yaml:409-413`) |
| 3 | Speech Content Editing | `content_edit` | `insert_before, insert_after, delete, delete_before, delete_after, replace` (`pe.config.yaml:170`) | ref speech | `content_scaled` (`pe.config.yaml:187-188`) |
| 4 | Lyric Editing | `vocal_edit` | replace (`orig`→`new`) | **a cappella** vocals (`COOKBOOK.md:188-190`) | `content_scaled` (`pe.config.yaml:526-527`); Cookbook omits `gen_seconds` |
| 5 | Pitch Editing | `pitch_edit` | inc/dec; 1/2/3 st (`pe.config.yaml:249-255`) | ref speech | `equal_length` |
| 6 | Speed Editing | `speed_edit` | 0.5/0.75/1.25/1.5/2.0× (`pe.config.yaml:211-215`) | ref speech | `speed_scaled` |
| 7 | Volume Editing | `volume_edit` | inc/dec; 5/10/15 dB (`pe.config.yaml:227-234`) | ref speech | `equal_length` |
| 8 | Emotion | `emotion_edit` | 8 classes (`pe.config.yaml:274-275`) | ref speech | `equal_length` (+PE multipliers `:305-316`) |
| 9 | Timbre | `voice_edit` | `timbre_desc` (`pe.config.yaml:439-461`) | ref speech | `equal_length` |
| 10 | De-accent | `accent_edit` | — (`pe.config.yaml:531-540`) | ref speech | `equal_length` |
| 11 | Nonverbal | `nonverbal_edit` | `delete/add_before/add_after/add_head/add_tail`; ~40 events (`pe.config.yaml:468,543-699`) | ref speech | `equal_length` (+PE `:476-490`) |
| 12 | Whisper | `whisper_edit` | `to_normal/to_whisper` (`pe.config.yaml:506`) | ref speech | `equal_length` |
| 13 | Speech Enhancement | `enhance_speech` | denoise/dereverb/enhance | ref noisy speech | `equal_length` |
| 14 | Speech Separation | `separate_speech` | `by_order`, `by_content` (`pe.config.yaml:341-358`) | ref mixture | `equal_length` |
| 15 | Music Separation | `extract_vocals` | `singing_only`, `all_human_voices` (`pe.config.yaml:359-374`) | ref **music mix** | `equal_length` |
| 16 | Target Speaker Extraction | `separate_speech/by_content` | keep speaker by `{text}` (`COOKBOOK.md:543-566`) | ref mixture | `equal_length` |
| 17 | Quality restoration | `improve_quality` | `bandwidth_extension`, `remove_effect` (`pe.config.yaml:375-395`) | ref degraded speech | `equal_length` |

### 1.2 Duration semantics

- VAE 480× downsample → 50 Hz latent; generated latent = `ceil(gen_seconds*24000/480)`;
  `gen_seconds=None` ⇒ regenerate a segment equal to source (`infer_auk.py:285-291`).
- Supplying an explicit `gen_seconds` equal to the actual source duration for an
  `equal_length` task is **valid** and is the preferred determinism choice; it is
  **not** an invalid input.
- Training window 0.3–30 s (`docs/FINETUNING.md:187`, `train.py:186-187`); ComfyUI
  source+target ≤ 30 s (`docs/COMFYUI.md:59-62`).
- VAD skip list: `enhance_speech, separate_speech, extract_vocals, improve_quality,
  nonverbal_edit, vocal_edit` and `whisper_edit/to_normal` (`pe.config.yaml:120`,
  `pe.py:1170-1175`).

---

## 2. Current Russian evaluation coverage (and its exact scope)

Five checkpoints on `local_tests/eval_pack/pack.json` (120 items, `pack_entries=120`,
seed 1234, nfe 64, cfg 2.0, trim false): `local_tests/{u0_control,s2_A_250,s2_A_500,
s2_B_250,s2_B_500}`. u0 control = `auk_ru_10000`. Blind set `local_tests/blind_s2/`
= 685 = 137 tasks × 5 variants (`capability 60, tool 175, tts 240, phonetics 80,
clone 125, phrase 5`).

Composition: `tts` 48 (RU zero-shot), `clone` 25 (RU), `tool` 35 (7 ops × 5),
`capability` 12. `phonetic_pack` 16 (RU ж/з, ч/ц, ы/и) and `phrase_training` 1 are
RU-specific, not upstream capabilities.

### 2.1 Corrected validity findings

1. **PE catalog vs raw generalization.** The tool pack uses 6 dB and 0.9×/1.1×, and
   `cap_speed_12` uses 1.2× — outside the PE enum. Since `run_eval_pack.py` calls the
   **raw** `AukInfer.generate` (PE bypassed), these are legitimate **off-template
   generalization** probes. They are not "invalid"; they simply do not measure the
   documented-template capability and should be reported separately from in-template
   items.
2. **Development vs training.** The 25 `clone` refs appear in
   `local_train/data_s2_full/val.jsonl` and the 35 tool refs in
   `local_train/data_s2_tools_v3/val.jsonl` (from `leakage_report.json`). Being in a
   *val* manifest is a **legitimate development-eval** status, **not** gradient
   training contamination. Split classification:
   - **train** (gradient): none of these ids (`local_train/data*`,
     `data_s2_full/train.jsonl`, `data_s2_tools_v3/train.jsonl`).
   - **monitored val/dev** (loss/sample monitoring, checkpoint selection):
     clone refs → `data_s2_full/val.jsonl`; tool refs → `data_s2_tools_v3/val.jsonl`.
   - **untouched final**: none configured.
   So they are a valid **dev** set, not a final heldout set.
3. **Speaker-cluster overlap is the real tools issue.** `../TOOLS_OVERLAP.md`:
   tools train/val source ids are disjoint (0), tools val vs s2 train **ids = 0**,
   but **speaker-cluster** overlaps exist — `tools_train_vs_s2_train_clusters = 2189`
   and **`tools_val_vs_s2_train_clusters = 317`**. So 317 tool-val clusters share
   speakers with s2 speech training, even though clip ids do not. Also
   `tools_val_vs_s2_val_clusters = 0`. This is the item to report, not id-level
   contamination.
4. **`cap_speech_edit` (replace 'привет'→'здравствуйте')** over `user_ref.wav`: ASR
   shows no 'привет' (`capability_probe/manifest.json:30`). Caveat: ASR is not proof
   of absence; this needs acoustic confirmation. The documented replace semantics is
   unverified for this source until then.
5. **`cap_vocal_extraction`** over `user_ref.wav`: single-speaker **speech-only**
   source, so music/singing separation has no target. This is a modality mismatch and
   is invalid for `extract_vocals` regardless of acoustic detail.
6. **`cap_deaccent` / `cap_enhancement` / `cap_quality`**: whether `user_ref.wav`
   carries a regional accent, or noise/reverb/telephone degradation, **cannot be
   proven by ASR nor by the absence of synthetic noise**. Marked
   **effect-eligibility unverified** until acoustic evidence (e.g. spectrum,
   reverberation estimate, listening) is produced. Not declared invalid.
7. **`cap_nonverbal_breath` `Add a breath before the last word`**: the anchor is
   implicit but may be semantically resolvable from the source; the PE template
   prefers an explicit anchor. Not intrinsically invalid; prefer an explicit
   recognized anchor for a scored fixture.
8. **No native-language original test** in the 120-item pack (all RU); original
   zh/en capability is unmeasured.
9. **Metric caveat**: `quality.recall` divides by unique expected words
   (`quality.py:51-57`) → can exceed 1 and return 1.0 on wrong output. Use
   `wer_metrics` (`quality.py:87-97`).

### 2.2 Missing original capabilities (no fixture anywhere)

`instruct_tts` (no-ref), `vocal_edit`, `voice_edit`, `separate_speech.by_order` &
`by_content`, `extract_vocals` on real music, `accent_edit` on accented audio,
`improve_quality.bandwidth_extension`/`remove_effect`, `whisper_edit.to_normal`,
content-edit insert/delete variants, emotions beyond happy/sad/angry, nonverbal
delete/add_head/add_tail, and any native zh/en task.

---

## 3. Metrics and controls (corrected)

- **Gemini audio judge** (root delegates it; local proxy only per `AGENTS.md`) is the
  requested perceptual judge, with **objective corroboration**. Human listening is
  **optional**, not a required blocker.
- **Objective magnitudes**: use the **sibling corrected** DSP
  `dsp_review_worker/dsp_core_v2.py` (+ `dsp_v2_controls.json`). The older
  `tool_dsp_measure.py` is superseded; do **not** use its SNR/preservation-correlation
  as a pass metric, and never treat **raw waveform correlation** as voice similarity
  or content correctness (it is neither).
- **Russian content**: `quality.wer_metrics`.
- **Speaker similarity**: `speaker_embed.py` (ONNX wespeaker cosine).
- **zh/en content**: no local ASR; use the Gemini judge + optional human.

Controls: `upstream_base` (original reference) vs `s1_u10000` (RU-adapted control);
every fixture run by all controls with identical instruction/seed/`gen_seconds`.

---

## 4. Heldout scope vocabulary (used in the manifest)

- `path_absent_from_finetune_manifests` — the exact source path is not present in the
  local fine-tune train/val JSONL (verified by `check_leakage.py`). Does **not** prove
  content or speaker novelty.
- `dev_monitored_val` — appears in a fine-tune *val* manifest (legitimate dev eval).
- `untouched_final` — held out from all manifests; none configured yet.
- `unknown_vs_upstream_pretrain` — cannot be verified locally; upstream pretraining
  exclusion is **not asserted**.

---

## 5. References

- `README.md:82-182`; `docs/COOKBOOK.md`; `src/auk/infer/pe.config.yaml`;
  `infer_auk.py:260-358`; `infer_cli.py:26-66`.
- `docs/FINETUNING.md:53,185-190,226-232`; `docs/COMFYUI.md:59-62`.
- `../full_sha256.json`, `../TOOLS_OVERLAP.md`, `dsp_review_worker/`.
- `local_tests/eval_pack/pack.json`, `phonetic_pack/pack.json`,
  `blind_s2/LISTEN.csv`, `capability_probe/manifest.json`.
- This folder: `assets_probe.json`, `leakage_report.json`, `probe_assets.py`,
  `check_leakage.py`.
