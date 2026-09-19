"""Regression tests for src/auk/infer/quality.py metric bugs (16.09.2026).

Run:  .venv\\Scripts\\python.exe local_train\\test_quality_metrics.py
Also pytest-compatible. Does not run ASR/GPU: transcribe is stubbed in pick_best test.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import auk.infer.quality as q  # noqa: E402


def test_recall_no_overcount():
    assert q.recall("да да нет", "да да нет") == 1.0


def test_recall_partial():
    v = q.recall("да да нет", "нет")
    assert abs(v - 1 / 3) < 1e-9, v


def test_recall_extra_words_not_rewarded():
    assert q.recall("да да нет", "да да нет нет нет") == 1.0


def test_recall_never_above_one():
    for exp, heard in [("один", "один один один"), ("раз два", "раз раз два два два")]:
        assert q.recall(exp, heard) <= 1.0


def test_recall_missing_word():
    assert q.recall("хлеб масло", "хлеб") == 0.5


def test_recall_wrong_word_zero():
    assert q.recall("хлеб", "хлеп") == 0.0


def test_recall_empty_expected_is_neutral():
    # documented behaviour: no Russian tokens in expected -> neutral 1.0 (language filter)
    assert q.recall("", "что угодно") == 1.0
    assert q.recall("hello world", "привет") == 1.0


def test_wer_metrics_substitution():
    m = q.wer_metrics("а б в", "а х в")
    assert m["subs"] == 1 and m["hits"] == 2
    assert abs(m["wer"] - 1 / 3) < 1e-9


def test_wer_metrics_insertion_deletion():
    assert q.wer_metrics("а б в", "а б в г")["ins"] == 1
    assert q.wer_metrics("а б в", "а б")["dels"] == 1


def test_clip_ratio():
    x = np.array([0.0, 0.5, 0.99, -0.995], dtype="float32")
    assert abs(q.clip_ratio(x) - 0.5) < 1e-9


def test_pick_best_prefers_exact_text():
    heard = {0: "", 1: "раз два три"}
    cands = [(torch.zeros(2400), 24000), (torch.randn(2400), 24000)]
    orig = q.transcribe
    q.transcribe = lambda wav, sr: heard[0 if float(wav.abs().sum()) == 0.0 else 1]
    try:
        idx = q.pick_best(cands, "раз два три")
    finally:
        q.transcribe = orig
    assert idx == 1, idx


def test_pick_best_uses_unrounded_wer():
    orig = q.score_candidate
    vals = {1: (0.304, 0.0, 0.0), 2: (0.301, 0.0, 0.0)}
    q.score_candidate = lambda wav, sr, text: vals[wav.shape[0]]
    try:
        idx = q.pick_best([(torch.zeros(1, 2400), 24000), (torch.zeros(2, 2400), 24000)], "x")
    finally:
        q.score_candidate = orig
    assert idx == 1, idx


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "->", e)
        except Exception as e:
            failed += 1
            print("ERROR", t.__name__, "->", type(e).__name__, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
