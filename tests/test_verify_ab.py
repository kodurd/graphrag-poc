"""Вердикт A/B verify-then-answer — чистая логика первичных гейтов."""

from __future__ import annotations

from eval.verify_ab import verdict


def _pair(a_rel, b_rel, *, a_abst=False, b_abst=False, residual=0.0, a_faith=0.5, b_faith=0.9):
    return {"a_rel": a_rel, "b_rel": b_rel, "a_abst": a_abst, "b_abst": b_abst,
            "residual": residual, "a_faith": a_faith, "b_faith": b_faith}


def test_ship_when_relevance_holds_no_abstention_cost():
    pairs = [_pair(0.8, 0.8, residual=0.1) for _ in range(6)]
    v = verdict(pairs)
    assert v["ship"] is True
    assert v["relevance_delta"] == 0.0  # falsy-zero: ровно 0 — валидно, не «нет данных»
    assert v["abstention_delta"] == 0.0


def test_no_ship_when_relevance_drops():
    pairs = [_pair(0.8, 0.5, residual=0.1) for _ in range(6)]  # дельта −0.3
    v = verdict(pairs)
    assert v["ship"] is False
    assert "relevance" in v["reason"]


def test_no_ship_when_abstention_balloons():
    # relevance держится на ответивших, но B массово воздерживается
    pairs = [_pair(0.8, 0.8, residual=0.1) for _ in range(8)]
    pairs += [_pair(0.8, None, b_abst=True) for _ in range(2)]  # 2/10 воздержаний = +20пп
    v = verdict(pairs)
    assert v["abstention_delta"] > 0.05
    assert v["ship"] is False and "воздержания" in v["reason"]


def test_no_ship_when_residual_above_norm():
    pairs = [_pair(0.8, 0.8, residual=0.5) for _ in range(6)]  # residual 0.5 > 0.2
    v = verdict(pairs)
    assert v["ship"] is False and "residual" in v["reason"]


def test_faithfulness_is_confirming_not_blocking():
    # faithfulness падает, но первичные гейты ок → всё равно ship (faith не блокирует)
    pairs = [_pair(0.8, 0.8, residual=0.1, a_faith=0.9, b_faith=0.4) for _ in range(6)]
    v = verdict(pairs)
    assert v["faith_delta"] < 0  # подтверждающий сигнал плохой
    assert v["ship"] is True  # но первичные гейты решают


def test_empty_pairs():
    v = verdict([])
    assert v["ship"] is False  # relevance_delta None → relevance_ok False
