"""A/B генератора: чистые агрегаты (compare_arms/mean_faith) и вердикт ship/no-ship."""

from __future__ import annotations

import pytest

from eval.gen_ab import compare_arms, mean_faith, verdict


# --- агрегаты ---

def test_mean_faith_ignores_none():
    assert mean_faith([1.0, 0.0, None, 0.5]) == pytest.approx(0.5)  # None (воздержание/сбой) вне среднего
    assert mean_faith([None, None]) is None
    assert mean_faith([]) is None


def test_compare_arms_positive_delta_when_strict_more_faithful():
    c = compare_arms(baseline_scores=[0.3, 0.5], strict_scores=[0.9, 1.0])
    assert c["baseline_mean"] == pytest.approx(0.4)
    assert c["strict_mean"] == pytest.approx(0.95)
    assert c["delta"] == pytest.approx(0.55)
    assert c["n_baseline"] == 2 and c["n_strict"] == 2


def test_compare_arms_delta_none_when_arm_empty():
    c = compare_arms(baseline_scores=[None], strict_scores=[0.9])
    assert c["baseline_mean"] is None and c["delta"] is None


# --- вердикт ---

def _pair(base, strict, *, base_abst=False, strict_abst=False):
    return {"base_score": base, "strict_score": strict,
            "base_abstained": base_abst, "strict_abstained": strict_abst}


def test_verdict_ship_when_strict_significantly_better_no_abstention_cost():
    """6 сильных парных дельт (+1.0) → p<=0.05, воздержания не растут → SHIP.

    Заодно falsy-zero: base_score ровно 0.0 — валидный балл, дельта считается."""
    pairs = [_pair(0.0, 1.0) for _ in range(6)]
    v = verdict(pairs)
    assert v["n_paired"] == 6
    assert v["faith_delta"] == pytest.approx(1.0)
    assert v["p"] <= 0.05
    assert v["abstention_delta"] == pytest.approx(0.0)
    assert v["ship"] is True


def test_verdict_no_ship_when_abstention_guardrail_breached():
    """Faithfulness значимо растёт, но strict массово воздерживается → NO-SHIP по guardrail."""
    pairs = [_pair(0.0, 1.0) for _ in range(6)]
    # +2 вопроса, где strict воздержался (score None), а baseline ответил → рост воздержаний
    pairs += [_pair(0.5, None, strict_abst=True) for _ in range(2)]
    v = verdict(pairs)
    assert v["faith_delta"] == pytest.approx(1.0)  # парные дельты — только 6 полных пар
    assert v["p"] <= 0.05
    assert v["abstention_delta"] > 0.05
    assert v["ship"] is False
    assert "guardrail" in v["reason"]


def test_verdict_no_ship_when_effect_not_significant():
    """Мало парных дельт → перестановочный тест не даёт значимости → NO-SHIP."""
    pairs = [_pair(0.0, 1.0) for _ in range(3)]  # p = 2/2^3 = 0.25
    v = verdict(pairs)
    assert v["p"] > 0.05
    assert v["ship"] is False
    assert "значим" in v["reason"]


def test_verdict_no_ship_when_no_paired_scores():
    """Нет совместно-оценённых пар → вердикт невозможен, NO-SHIP."""
    v = verdict([_pair(None, 1.0), _pair(0.5, None, strict_abst=True)])
    assert v["n_paired"] == 0
    assert v["ship"] is False
    assert "нет совместно" in v["reason"]


def test_verdict_empty_pairs():
    v = verdict([])
    assert v["ship"] is False and v["n_paired"] == 0
