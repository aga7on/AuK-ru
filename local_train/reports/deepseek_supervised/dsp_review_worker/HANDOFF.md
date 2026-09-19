# HANDOFF — DSP correction worker (v2)

Owner: DeepSeek v4.1 Flash DSP correction worker. Scope: **only** this directory.
No shared scripts/docs/source modified; no training/synthesis/network; `.venv` Python.

## Artifacts

| file | what |
|---|---|
| `dsp_core_v2.py` | corrected independent measurement core (replaces logic of `local_train/tool_dsp_measure.py`) |
| `dsp_measure_v2.py` | runs 175 actual tool WAVs + controls, writes the JSON/MD below |
| `dsp_controls_v2.py` | focused positive/negative control tests with explicit assertions |
| `dsp_v2_perfile.json` | 175 records (7 ops × 5 tasks × 5 variants) with levels, hashes, alignment, metrics |
| `dsp_v2_summary.json` | per op×variant: n/valid/invalid, invalid reasons, effect distribution, prospective thresholds |
| `dsp_v2_inputs.json` | every source/output path + full sha256, and clean-provenance for noise_add |
| `dsp_v2_controls.json` | control observations + assertion results |
| `DSP_REVIEW_V2.md` | concise review for root |
| `controls/*.wav` | FLOAT control signals (primitive transforms only) |

## Run

```powershell
.venv\Scripts\python.exe local_train\reports\deepseek_supervised\dsp_review_worker\dsp_controls_v2.py
.venv\Scripts\python.exe local_train\reports\deepseek_supervised\dsp_review_worker\dsp_measure_v2.py
# optional content evidence once a verified mapping exists:
#   ... dsp_measure_v2.py --content-evidence <verified_map.json>   (key "variant/task_id")
```
Bound: 175 files ≈ 51 s; controls 16/16 asserted.

## Mapping of root's six blocking defects

1. **Activity floor** — absolute dBFS floor + p95-relative floor; all-zero/non-finite/too-short
   are hard-invalid; absolute active RMS is logged. This exposed `tool_pitch_down_0`
   (all variants ≈ −66 dBFS) which the old script scored as valid.
2. **Segmental SNR** — only after explicit limited-lag (±300 ms) alignment + envelope-correlation
   applicability gate; otherwise `inconclusive`. Added phase-robust noise-floor reduction.
   SR mismatch → explicit resample + `resampled` flag (no silent truncation). Silence never
   counted as perfect denoise.
3. **Volume** — FLOAT WAV with headroom; median/IQR frame gain, peak, clip ratio, magnitude
   trustworthiness; positive *and* negative controls asserted.
4. **Speed/pitch** — speed = active-**span** proxy gated by stretch-invariant content
   comparability (MFCC NN + DTW) → invalid when not comparable. Pitch = paired voiced-frame
   median + coverage + octave guard.
5. **Clean path** — resolved from provenance (`dataset noise_add_val_<id>.wav`, verified equal to
   the kyutai clip), paths + sha256 logged.
6. **Reporting** — n/valid/invalid per op×variant, distribution, prospective thresholds; no
   headline pass from a mean.

## What the corrected measurement shows (facts)

- **volume_up**: requested +6 dB, median effect 0.9–2.6 dB; only 0–2 of 5 fall in (3,9) dB.
- **volume_down**: requested −6 dB; correct direction only 1–2 of 5; worst `tool_volume_down_4`
  (e.g. B@500 +0.97 dB, i.e. louder). Old mean-of-valid (~+0.4 dB) hid this.
- **speed_down**: content-comparable only for B@500 (5/5 valid); u10000/A/B@250 are
  non-comparable → invalid rather than a bogus rate.
- **speed_up**: B@500 is the only 5/5 valid.
- **pitch_up**: consistent ≈ +2 st everywhere. **pitch_down**: 3/5 valid (one near-silent task,
  one source with almost no voiced frames).
- **noise_add**: segmental-SNR applicable in only 3–4 of 5; where applicable the SNR gain is
  negative (metric penalises re-synthesis), while the phase-robust noise-floor reduction is
  positive (median 8–13 dB). No denoise verdict is asserted.

## Not concluded / open for root

- No per-file variant→judge mapping exists in-repo without opening `blind_s2/SECRET_map.csv`
  (forbidden until listening ends), so ASR/Gemini content evidence was **not** attached.
  `--content-evidence` is ready for a verified mapping.
- Denoise is reported as weak/inconclusive by design — unsuitable ground truth for a conclusive
  claim on regenerated audio.
- No winner between u10000 / A@250 / A@500 / B@250 / B@500 is declared here; this is an
  objective magnitude/validity measurement, decision stays with the blind protocol.
