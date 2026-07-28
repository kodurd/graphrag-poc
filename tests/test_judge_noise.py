"""Чистое ядро диагностики шума судьи: отбор групп, стабильность, вердикт."""

from __future__ import annotations

import pytest

from eval.judge_noise import group_stats, select_groups, stability, verdict


def _rec(faith):
    return {"metrics": {"faithfulness": faith}}


# --- select_groups ---

def test_select_groups_bins_by_faithfulness():
    recs = [_rec(0.0), _rec(0.1), _rec(0.9), _rec(1.0), _rec(0.5), _rec(None)]
    g = select_groups(recs)
    assert len(g["low"]) == 2   # 0.0, 0.1 (проверка границ групп low/high/mid)
    assert len(g["high"]) == 2  # 0.9, 1.0
    assert len(g["mid"]) == 1   # 0.5
    # None-запись отброшена (нет faithfulness)
    assert len(g["low"]) + len(g["high"]) + len(g["mid"]) == 5


def test_select_groups_mid_empty_on_bimodal_data():
    """Как в реальном снимке n=96: нет записей в 0.2–0.8 → mid пуст, не ошибка."""
    g = select_groups([_rec(0.0), _rec(0.1), _rec(0.9), _rec(1.0)])
    assert g["mid"] == []


# --- stability ---

def test_stability_stable_zero_variance():
    s = stability([0.0, 0.0, 0.0, 0.0, 0.0])
    assert s["stable"] is True
    assert s["variance"] == pytest.approx(0.0)
    assert s["mean"] == pytest.approx(0.0)
    assert s["n_scored"] == 5 and s["no_score"] is False


def test_stability_noisy_high_variance():
    s = stability([0.0, 1.0, 0.0, 1.0, 0.0])
    assert s["stable"] is False  # оценки в разных корзинах
    assert s["variance"] > 0.2
    assert s["mean"] == pytest.approx(0.4)


def test_stability_none_counted_but_not_crashing():
    """None (воздержание/сбой) не роняет расчёт, учитывается в числе не-None."""
    s = stability([0.0, None, 0.0, None, 0.0])
    assert s["n_scored"] == 3
    assert s["mean"] == pytest.approx(0.0)
    assert s["no_score"] is False


def test_stability_all_none_is_no_score():
    s = stability([None, None, None, None, None])
    assert s["no_score"] is True
    assert s["mean"] is None
    assert s["stable"] is False
    assert s["n_scored"] == 0


# --- group_stats ---

def test_group_stats_noisy_fraction_and_variance():
    per_record = [
        stability([0.0, 0.0, 0.0]),   # stable
        stability([0.0, 1.0, 0.0]),   # noisy
        stability([1.0, 1.0, 1.0]),   # stable
        stability([0.0, 0.5, 1.0]),   # noisy
    ]
    gs = group_stats(per_record)
    assert gs["n_scored"] == 4
    assert gs["noisy_frac"] == pytest.approx(0.5)  # 2 из 4
    assert gs["mean_variance"] > 0


def test_group_stats_excludes_no_score_from_denominator():
    per_record = [
        stability([0.0, 0.0, 0.0]),          # stable
        stability([0.0, 1.0, 0.0]),          # noisy
        stability([None, None, None]),        # no_score — вне знаменателя
    ]
    gs = group_stats(per_record)
    assert gs["n_scored"] == 2 and gs["no_score"] == 1
    assert gs["noisy_frac"] == pytest.approx(0.5)  # 1 шумная из 2 оценённых, no_score не в знаменателе


def test_group_stats_empty_group_is_na():
    gs = group_stats([])
    assert gs["n_scored"] == 0 and gs["noisy_frac"] is None
    assert gs.get("note") == "n/a"


def test_group_stats_all_no_score_is_na():
    gs = group_stats([stability([None, None]), stability([None])])
    assert gs["n_scored"] == 0 and gs["noisy_frac"] is None


# --- verdict ---

def _low_stats(noisy_frac, mean_faith, n_scored=20):
    """Собирает low_stats-структуру напрямую для проверки вердикта."""
    return {"n": n_scored, "n_scored": n_scored, "no_score": 0,
            "noisy_frac": noisy_frac, "mean_variance": 0.1, "mean_faith": mean_faith}


def test_verdict_fix_by_noisy_gate():
    v = verdict(_low_stats(noisy_frac=0.5, mean_faith=0.05))
    assert v["fix_warranted"] is True and v["gate_noisy"] is True


def test_verdict_fix_by_pole_lift_gate():
    """Полюс «0» пере-усреднился до 0.45 (≥0.30) → систематическое занижение → fix."""
    v = verdict(_low_stats(noisy_frac=0.1, mean_faith=0.45))
    assert v["fix_warranted"] is True and v["gate_lift"] is True


def test_verdict_not_warranted_when_stable_and_low():
    v = verdict(_low_stats(noisy_frac=0.1, mean_faith=0.05))
    assert v["fix_warranted"] is False
    assert "честны" in v["reason"]


def test_verdict_falsy_zero_low_mean_not_treated_as_missing():
    """low_mean ровно 0.0 — валидное число: вердикт «не warranted» по верной причине."""
    v = verdict(_low_stats(noisy_frac=0.1, mean_faith=0.0))
    assert v["fix_warranted"] is False
    assert v["low_mean"] == 0.0  # не None, не «нет данных»


def test_verdict_no_scored_records():
    v = verdict(group_stats([]))
    assert v["fix_warranted"] is False
    assert "невозможен" in v["reason"]
