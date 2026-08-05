"""judge_calibration: MAE, корреляция, консистентность, вердикт-через-gold."""

from __future__ import annotations

import pytest

from eval.judge_calibration import (
    calibration,
    consistency,
    mae,
    pearson,
    refusal_verdict,
)


def test_mae_ignores_none():
    assert mae([1.0, 0.0, None], [1.0, 0.5, 0.5]) == 0.25  # |1-1|=0, |0-.5|=.5 -> mean 0.25


def test_pearson_perfect_and_none():
    assert pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(1.0)
    assert pearson([1.0], [1.0]) is None  # n<2


def _cross(sid, ds, qw, answer="ok", abst=False):
    return {"source_id": sid, "question": f"q-{sid}", "answer": answer,
            "abstained": {"faithfulness": abst},
            "deepseek": ds, "qwen": qw}


def test_consistency_agreement_bands():
    cross = [
        _cross("t1", {"faithfulness": 1.0}, {"faithfulness": 1.0}),   # Δ=0
        _cross("t2", {"faithfulness": 1.0}, {"faithfulness": 0.5}),   # Δ=0.5
    ]
    c = consistency(cross)["faithfulness"]
    assert c["n"] == 2
    assert c["agree"]["|d|<=0.1"] == 0.5   # только t1
    assert c["agree"]["|d|<=0.3"] == 0.5


def test_calibration_mae_to_gold():
    cross = [_cross("t1", {"answer_relevance": 0.0}, {"answer_relevance": 1.0})]
    gold = {"t1": {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}}
    cal = calibration(cross, gold)
    assert cal["answer_relevance"]["deepseek_mae"] == 1.0  # |0-1|
    assert cal["answer_relevance"]["qwen_mae"] == 0.0      # |1-1|


def test_refusal_verdict_uses_gold_not_agreement():
    """На отказе Qwen ближе к человеку -> вердикт: DeepSeek занижал."""
    cross = [_cross("t1", {"answer_relevance": 0.0}, {"answer_relevance": 0.9},
                    answer="в источнике недостаточно данных", abst=True)]
    gold = {"t1": {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}}
    rv = refusal_verdict(cross, gold)
    assert rv["n_refusal_gold"] == 1
    assert rv["qwen_relevance_mae"] < rv["deepseek_relevance_mae"]
    assert "занижал" in rv["verdict"]
