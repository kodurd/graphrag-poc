"""U4: калибровка порога — кандидатные пороги (>0) + guardrail-выбор."""

from __future__ import annotations

import pytest

from eval.threshold_calib import (
    candidate_thresholds,
    positive_scores,
    recommend_threshold,
)


# --- кандидатные пороги только из положительных логитов ---

def test_positive_scores_filters_non_positive():
    assert positive_scores([-1.0, 0.0, 0.5, 2.0]) == [0.5, 2.0]


def test_candidate_thresholds_all_positive():
    cands = candidate_thresholds([-2.0, 0.0, 0.5, 1.0, 3.0])
    assert cands and all(t > 0 for t in cands)


def test_candidate_thresholds_empty_when_no_positive():
    # все скоры <= 0 -> порог невыразим при семантике filter_by_threshold (min<=0 = выкл)
    assert candidate_thresholds([-1.0, 0.0, -0.3]) == []


# --- guardrail-выбор ---

def test_recommend_picks_precision_raising_abstention_safe():
    base = {"precision": 0.40, "abstention": 0.60}
    evals = [
        {"threshold": 0.5, "precision": 0.55, "abstention": 0.60},   # precision вверх, воздержания те же
        {"threshold": 1.0, "precision": 0.50, "abstention": 0.55},
    ]
    rec = recommend_threshold(base, evals)
    assert rec["on"] is True and rec["threshold"] == 0.5  # лучший precision среди подходящих


def test_recommend_rejects_when_abstention_rises():
    base = {"precision": 0.40, "abstention": 0.60}
    evals = [{"threshold": 0.9, "precision": 0.80, "abstention": 0.70}]  # precision вверх, но воздержания растут
    rec = recommend_threshold(base, evals)
    assert rec["on"] is False and rec["threshold"] == 0.0


def test_recommend_rejects_when_no_precision_gain():
    base = {"precision": 0.40, "abstention": 0.60}
    evals = [{"threshold": 0.5, "precision": 0.40, "abstention": 0.55}]  # precision не вырос
    rec = recommend_threshold(base, evals)
    assert rec["on"] is False and rec["threshold"] == 0.0


def test_recommend_off_on_empty_candidates():
    rec = recommend_threshold({"precision": 0.4, "abstention": 0.6}, [])
    assert rec["on"] is False and rec["threshold"] == 0.0
