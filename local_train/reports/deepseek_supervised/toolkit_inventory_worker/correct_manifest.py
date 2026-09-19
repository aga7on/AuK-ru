"""Correct factual claims in eval_manifest_proposal.json (in place, idempotent-ish).

Applies the root-requested corrections:
- full sha256 (from ../full_sha256.json) instead of the self-contradictory "full ... first2GiB".
- heldout scope relabelled: path-absence from manifests, dev/monitored-val, untouched-final, unknown-vs-pretrain.
- PE catalog != raw-model capability: off-template items not invalid.
- effect eligibility unverified unless acoustic evidence.
- no "gen_seconds provided" invalid rule; explicit source-length gen_seconds is valid.
- Gemini judge + objective corroboration instead of "2 listeners".
- corrected instruct-TTS language/duration; add an English short no-ref fixture.
- reclassify current clone/tool as dev/monitored-val; separate cluster overlap (TOOLS_OVERLAP.md).
- add the authorized 3-fixture / 3-control diagnostic block.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "eval_manifest_proposal.json")
FULL = os.path.join(HERE, "..", "full_sha256.json")

FULL_SHA = json.load(open(FULL, encoding="utf-8"))

CONTROLS = {
    "upstream_base": {
        "path": "ckpts/AuK/auk_base.safetensors",
        "role": "original capability reference",
        "size_bytes": 6122209092,
        "sha256_full": FULL_SHA["upstream_base"]["sha256"],
        "sha256_first_2GiB_partial": "cf5a4ec2530eb714946bfc715c3959492c759bef98ae3cf639c4222b5d4de873",
        "hash_note": "partial value is sha256 of the first 2 GiB only; use sha256_full for identity",
    },
    "s1_u10000": {
        "path": "local_train/run_ru_s1/merged/auk_ru_10000.safetensors",
        "role": "Russian-adapted control (BASELINE.md:31-35)",
        "size_bytes": 6122209092,
        "sha256_full": FULL_SHA["s1_merged_10000"]["sha256"],
        "sha256_first_2GiB_partial": "4f3eca357e96c273f473cba4bbc7b0abb77c09f1fadaa7a9db3494c631c68ee0",
        "hash_note": "partial value is sha256 of the first 2 GiB only",
    },
    "s2_B500": {
        "path": "local_train/run_s2_B/merged/auk_s2_B_500.safetensors",
        "role": "s2 B@500 adapted control",
        "size_bytes": 6122209092,
        "sha256_full": FULL_SHA["s2_B500"]["sha256"],
    },
}

METRIC_TOOLING = {
    "perceptual_judge": "Gemini audio judge via local proxy (root delegates judging; human listening OPTIONAL, not a blocker)",
    "objective_magnitude": "sibling corrected DSP: local_train/reports/deepseek_supervised/dsp_review_worker/dsp_core_v2.py; the older tool_dsp_measure.py is SUPERSEDED (do not use its SNR/preservation corr as a pass metric)",
    "corr_warning": "raw waveform correlation is NOT voice similarity and NOT content correctness; do not use it as such",
    "russian_content": "src/auk/infer/quality.py wer_metrics (NOT recall)",
    "speaker_similarity": "local_train/speaker_embed.py (wespeaker ONNX cosine)",
    "zh_en_content": "no local ASR; Gemini judge + optional human",
}

BROAD_REPLACE = {
    "2/2 listeners": "the Gemini judge (objective corroboration where available)",
    "both listeners": "the Gemini judge",
    "(2 listeners)": "(Gemini judge)",
    "2 listeners": "the Gemini judge",
    "for >=2/2 listeners": "by the Gemini judge",
    "for 2/2 listeners": "by the Gemini judge",
}

HELDOUT_RELABEL = {
    "verified_not_in_local_finetune": "path_absent_from_finetune_manifests",
    "verified_heldout": "path_absent_from_finetune_manifests",
}
HELDOUT_NOTE = (
    " Path-absence does not prove content/speaker novelty; overlap with upstream "
    "pretraining is unknown and not asserted."
)


def fix_text(s):
    if not isinstance(s, str):
        return s
    for a, b in BROAD_REPLACE.items():
        s = s.replace(a, b)
    return s


def walk_fix(obj):
    if isinstance(obj, dict):
        return {k: walk_fix(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk_fix(v) for v in obj]
    return fix_text(obj)


def main():
    d = json.load(open(MANIFEST, encoding="utf-8"))
    d = walk_fix(d)
    d["controls"] = CONTROLS
    d["metric_tooling"] = METRIC_TOOLING

    # heldout relabel + evidence note
    for f in d["fixtures"]:
        h = f.get("heldout")
        if isinstance(h, dict):
            st = h.get("status")
            if st in HELDOUT_RELABEL:
                h["status"] = HELDOUT_RELABEL[st]
            if "unknown_vs_upstream_pretrain" not in h.get("evidence", ""):
                h["evidence"] = h.get("evidence", "") + HELDOUT_NOTE
        # remove the equal-length "gen_seconds provided" invalid rule
        rules = f.get("invalid_input_rules") or []
        f["invalid_input_rules"] = [
            r for r in rules if "equal-length" not in r.lower() and "gen_seconds provided" not in r.lower()
        ]

    by_id = {f["id"]: f for f in d["fixtures"]}

    # metric tool fixes for objective fixtures
    for fid in ("smoke_upstream_pitch_up2", "smoke_ru_pitch_up2_heldout_ref"):
        by_id[fid]["metric"]["tool"] = (
            "dsp_review_worker/dsp_core_v2.py (supersedes tool_dsp_measure.py for magnitude)"
        )
        by_id[fid]["heldout"]["status"] = "path_absent_from_finetune_manifests"
    for fid in ("orig_speed_edit_150", "orig_volume_edit_up10",
                "expand_separate_speech_by_order_synthetic",
                "expand_target_speaker_extraction_synthetic"):
        if fid in by_id:
            by_id[fid]["metric"]["tool"] = by_id[fid]["metric"]["tool"].replace(
                "local_train/tool_dsp_measure.py", "dsp_review_worker/dsp_core_v2.py"
            ).replace("tool_dsp_measure.py", "dsp_review_worker/dsp_core_v2.py")

    # instruct TTS correction
    it = by_id["orig_instruct_tts_zh_noref"]
    it["language"] = "zh"
    it["instruction_language"] = "zh"
    it["text_language"] = "en"
    it["gen_seconds"] = 5.0
    it["duration_rule"] = (
        "text-only => explicit gen_seconds; target text is English (text_language=en). "
        "5.0 s for the 10-word sentence at a slow/gentle pace. Do NOT claim Chinese-output coverage."
    )
    it["expected_target_rule"] = (
        "intelligible ENGLISH rendering of 'Welcome home, how was work today?' (~7 words) at a "
        "slow, gentle pace, ~4.5-5.0 s; style female/gentle/soft"
    )
    it["note"] = (
        "Corrected: the original draft labelled this zh and used 1.7 s, inconsistent with the "
        "slow/gentle style and English target text."
    )

    # new English short no-ref instruct fixture
    en_instr = {
        "id": "orig_instruct_tts_en_noref_short",
        "capability": "Speech Generation: Instruct TTS",
        "task_type": "instruct_tts",
        "subtype": None,
        "language": "en",
        "instruction_language": "en",
        "text_language": "en",
        "instruction": ("Based on the following description: \"A calm, clear adult female narrator "
                        "with a warm, measured delivery and a slight smile in her voice, natural pace, "
                        "crisp articulation.\", generate speech content \"The ancient clockmaker wound "
                        "the brass key three times before sunrise.\"."),
        "source_path": None,
        "source_provenance": "text-only (no media)",
        "input_modality": "text_only_no_audio",
        "gen_seconds": 5.0,
        "duration_rule": "no ref => explicit gen_seconds; 10-word sentence, clear 4-5 s request",
        "reference_control": {"expected_ref_property": "none", "control_check": "no type=audio item"},
        "expected_target_rule": "intelligible English, calm clear female narrator, ~4.5-5.0 s",
        "heldout": {"status": "path_absent_from_finetune_manifests", "evidence": "text-only"},
        "metric": {"name": "gemini_content_style_naturalness", "tool": "Gemini judge + optional human",
                   "absolute_criterion": "content correct AND natural/gender/style consistent",
                   "paired_criterion": "s1_u10000 and s2_B500 not worse than upstream_base"},
        "invalid_input_rules": ["empty style/text", "gen_seconds unset or <=0", "audio item present would make it zero_shot_tts"],
        "status": "valid",
    }
    d["fixtures"].append(en_instr)
    by_id[en_instr["id"]] = en_instr

    # nonverbal implicit anchor
    nb = by_id["orig_nonverbal_remove_hum_en"]  # keep
    # add explicit-anchor note to add fixture
    if "orig_nonverbal_add_cough_en" in by_id:
        by_id["orig_nonverbal_add_cough_en"]["note"] = (
            "Anchor 'We tested' is explicit. For the current pack's 'before the last word' item the "
            "anchor is implicit and may be resolvable from the source; PE template prefers an explicit "
            "anchor (pe.config.yaml:494). Prefer explicit anchors for scored fixtures."
        )

    # reclassify current entries
    if "current_cap_speech_edit_invalid" in by_id:
        f = by_id["current_cap_speech_edit_invalid"]
        f["id"] = "current_cap_speech_edit_unverified"
        f["heldout"]["status"] = "unverified_source_absence"
        f["heldout"]["evidence"] = (
            "ASR of user_ref.wav shows no 'привет' (capability_probe/manifest.json:30); ASR is not "
            "proof of absence -- needs acoustic confirmation before scoring."
        )
        f["status"] = "unverified_effect_eligibility"
    if "current_cap_vocal_extraction_invalid" in by_id:
        f = by_id["current_cap_vocal_extraction_invalid"]
        f["id"] = "current_cap_vocal_extraction_modality_invalid"
        f["status"] = "invalid_current"
        f["heldout"]["evidence"] = "single-speaker speech source => modality mismatch for music separation"

    # new unverified / off-template current entries
    new_current = [
        {"id": "current_cap_deaccent_effect_eligibility_unverified",
         "capability": "Paralinguistic: De-accent", "task_type": "accent_edit", "subtype": None,
         "language": "en", "instruction": "Remove any accent from this speech, keep the same voice.",
         "source_path": "local_tests/user_ref.wav", "input_modality": "ref_audio+instruction",
         "heldout": {"status": "unverified_effect_eligibility",
                     "evidence": "whether the source carries a regional accent is not provable by ASR or by absence of synthetic noise; needs acoustic evidence"},
         "metric": {"name": "gemini_deaccent", "tool": "Gemini judge", "absolute_criterion": "n/a until eligibility verified", "paired_criterion": "n/a"},
         "status": "unverified_effect_eligibility"},
        {"id": "current_cap_enhancement_effect_eligibility_unverified",
         "capability": "Enhancement: Speech Enhancement", "task_type": "enhance_speech", "subtype": None,
         "language": "en", "instruction": "Remove the background noise and make the voice cleaner",
         "source_path": "local_tests/user_ref.wav", "input_modality": "ref_audio+instruction",
         "heldout": {"status": "unverified_effect_eligibility",
                     "evidence": "cleanliness of the source is not provable by ASR or by absence of synthetic noise; needs acoustic evidence"},
         "metric": {"name": "gemini_enhancement", "tool": "Gemini judge + corrected DSP if a clean reference exists", "absolute_criterion": "n/a until eligibility verified", "paired_criterion": "n/a"},
         "status": "unverified_effect_eligibility"},
        {"id": "current_cap_quality_effect_eligibility_unverified",
         "capability": "Enhancement: Quality restoration", "task_type": "improve_quality", "subtype": None,
         "language": "en", "instruction": "Improve the audio quality and make it clearer",
         "source_path": "local_tests/user_ref.wav", "input_modality": "ref_audio+instruction",
         "heldout": {"status": "unverified_effect_eligibility",
                     "evidence": "presence of a quality defect is not provable by ASR or absence of synthetic noise; needs acoustic evidence"},
         "metric": {"name": "gemini_quality", "tool": "Gemini judge + corrected DSP", "absolute_criterion": "n/a until eligibility verified", "paired_criterion": "n/a"},
         "status": "unverified_effect_eligibility"},
        {"id": "current_cap_nonverbal_breath_implicit_anchor",
         "capability": "Paralinguistic: Nonverbal", "task_type": "nonverbal_edit", "subtype": "add_before",
         "language": "en", "instruction": "Add a breath before the last word",
         "source_path": "local_tests/user_ref.wav", "input_modality": "ref_audio+instruction",
         "heldout": {"status": "valid_but_anchor_implicit",
                     "evidence": "'last word' may be resolvable from the source; PE template prefers an explicit recognized anchor (pe.config.yaml:494)"},
         "metric": {"name": "gemini_nonverbal", "tool": "Gemini judge", "absolute_criterion": "breath inserted at a defensible boundary; content preserved", "paired_criterion": "controls paired"},
         "status": "valid_but_anchor_implicit"},
        {"id": "current_cap_speed_12_off_template",
         "capability": "Acoustic Editing: Speed (off-template generalization)",
         "task_type": "speed_edit", "subtype": None, "language": "en",
         "instruction": "Change the speech speed to 1.2 times.",
         "source_path": "local_tests/user_ref.wav", "input_modality": "ref_audio+instruction",
         "heldout": {"status": "off_template_generalization",
                     "evidence": "1.2x is outside the PE enum (0.5/0.75/1.25/1.5/2.0); raw AukInfer has no enum validation, so this tests generalization, not an invalid task"},
         "metric": {"name": "speed_rate", "tool": "dsp_review_worker/dsp_core_v2.py", "absolute_criterion": "rate ~1.2 (report, no template gate)", "paired_criterion": "controls paired"},
         "status": "off_template_generalization"},
    ]
    d["fixtures"].extend(new_current)
    for f in new_current:
        by_id[f["id"]] = f

    # reclassify clone/tool current entries as dev/monitored-val
    for old, new, status in (
        ("current_clone_refs_contaminated", "current_clone_refs_dev_monitored", "dev_monitored_val"),
        ("current_tool_refs_contaminated", "current_tool_refs_dev_monitored_off_template", "dev_monitored_val"),
    ):
        if old in by_id:
            f = by_id[old]
            f["id"] = new
            f["status"] = status
    if "current_clone_refs_dev_monitored" in by_id:
        f = by_id["current_clone_refs_dev_monitored"]
        f["heldout"]["status"] = "dev_monitored_val"
        f["heldout"]["evidence"] = (
            "refs appear in local_train/data_s2_full/val.jsonl = legitimate development eval, NOT gradient "
            "contamination (not in any train.jsonl). Not an untouched final set. unknown_vs_upstream_pretrain."
        )
    if "current_tool_refs_dev_monitored_off_template" in by_id:
        f = by_id["current_tool_refs_dev_monitored_off_template"]
        f["heldout"]["status"] = "dev_monitored_val"
        f["heldout"]["evidence"] = (
            "refs appear in local_train/data_s2_tools_v3/val.jsonl = dev eval, not gradient contamination. "
            "TOOLS_OVERLAP.md: tools_val vs s2 train ids = 0 but speaker-cluster overlap = 317 (tools val vs "
            "s2 val = 0). Off-template operands (6 dB, 0.9x/1.1x). unknown_vs_upstream_pretrain."
        )

    # authorized diagnostic block
    d["authorized_diagnostic_synthesis"] = {
        "authorized_by": "root",
        "scope": "SMALL diagnostic only: 3 fixtures x 3 controls = 9 WAVs; no judge calls; no expansion to 27",
        "controls": ["upstream_base", "s1_u10000", "s2_B500"],
        "run_config": {"seed": 1234, "nfe": 64, "cfg_strength": 2.0, "trim": False, "bestofn": 1,
                       "no_clipping_normalization": True, "no_silence_trim": True, "one_engine_at_a_time": True},
        "fixtures": [
            {"id": "diag_pitch_en_up2", "language": "en", "task": "pitch_edit",
             "instruction": "Raise the pitch by 2 semitones.",
             "source_path": "assets/demo-input-audio/pitch/pitch-1-input.wav",
             "source_sha256": "e3ce8f28ff59246d328ac2ed44f4cf0ba20492228dad1ac53f63743378be6d47",
             "source_duration_s": 5.5, "gen_seconds": 5.5,
             "gen_seconds_note": "explicit value equals the actual source duration; valid for equal_length"},
            {"id": "diag_zeroshot_en", "language": "en", "task": "zero_shot_tts",
             "instruction": "Say the following with the same voice: 'The northern lights shimmered above the frozen lake while we watched in silence.'",
             "source_path": "assets/demo-input-audio/zero-shot-tts/ref.wav",
             "source_sha256": "ce1f449813c5769b6a4790fd50dac98a74a696795ee6b82c35068a11c147e79a",
             "source_duration_s": 9.95, "gen_seconds": 7.0,
             "gen_seconds_note": "sufficient duration for a 13-word sentence, no reference-length echo"},
            {"id": "diag_instructtts_en_noref", "language": "en", "task": "instruct_tts",
             "instruction": "Based on the following description: \"A calm, clear adult female narrator with a warm, measured delivery and a slight smile in her voice, speaking at a natural pace with crisp articulation.\", generate speech content \"The ancient clockmaker wound the brass key three times before sunrise.\".",
             "source_path": None, "gen_seconds": 5.0,
             "gen_seconds_note": "clear 4-5 s target for a 10-word novel sentence; previously untested no-ref path"},
        ],
    }

    json.dump(d, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("corrected fixtures:", len(d["fixtures"]))
    print("statuses:", {s: sum(1 for f in d["fixtures"] if f.get("status") == s) for s in set(f.get("status") for f in d["fixtures"])})


if __name__ == "__main__":
    main()
