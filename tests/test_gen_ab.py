"""A/B генератора: baseline vs claim-conservative по faithfulness — чистые агрегаты."""

from __future__ import annotations

import pytest

from eval.gen_ab import compare_arms, mean_faith


def test_mean_faith_ignores_none():
    assert mean_faith([1.0, 0.0, None, 0.5]) == pytest.approx(0.5)  # None (воздержание/сбой) вне среднего
    assert mean_faith([None, None]) is None
    assert mean_faith([]) is None


def test_compare_arms_positive_delta_when_strict_more_faithful():
    c = compare_arms(baseline_scores=[0.3, 0.5], strict_scores=[0.9, 1.0])
    assert c["baseline_mean"] == pytest.approx(0.4)
    assert c["strict_mean"] == pytest.approx(0.95)
    assert c["delta"] == pytest.approx(0.55)  # strict вернее → положительная дельта
    assert c["n_baseline"] == 2 and c["n_strict"] == 2


def test_compare_arms_delta_none_when_arm_empty():
    c = compare_arms(baseline_scores=[None], strict_scores=[0.9])
    assert c["baseline_mean"] is None and c["delta"] is None  # нет базлайна → дельта неопределена


def test_compare_arms_counts_scored_only():
    c = compare_arms(baseline_scores=[0.5, None, 0.7], strict_scores=[1.0])
    assert c["n_baseline"] == 2 and c["n_strict"] == 1
