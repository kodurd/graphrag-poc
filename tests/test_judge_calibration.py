"""judge_calibration: MAE, корреляция, консистентность, вердикт-через-gold."""

from __future__ import annotations

import pytest

from eval.judge_calibration import (
    calibration,
    calibration_split,
    consistency,
    mae,
    pearson,
    refusal_verdict,
    trusted_baseline,
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


_TRIPLE = ("faithfulness", "answer_relevance", "context_precision")


def _full(sid, ds, qw, *, route="mixed", faith_abst=False, abst=False):
    return {"source_id": sid, "question": f"q-{sid}", "route": route,
            "abstained": {"faithfulness": abst}, "qwen_faith_abstained": faith_abst,
            "deepseek": ds, "qwen": qw}


def test_trusted_baseline_common_set_excludes_qwen_none():
    """faith сравнивается только там, где ОБА не-None; Qwen-None -> в воздержания, не в контраст."""
    d = {"faithfulness": 1.0, "answer_relevance": 0.8, "context_precision": 0.7}
    cross = [
        _full("t1", d, {"faithfulness": 0.9, "answer_relevance": 0.8, "context_precision": 0.7}),
        _full("t2", d, {"faithfulness": None, "answer_relevance": 0.8, "context_precision": 0.7},
              faith_abst=True),  # Qwen воздержался по faith
    ]
    tb = trusted_baseline(cross)
    assert tb["by_metric"]["faithfulness"]["common_n"] == 1          # только t1
    assert tb["by_metric"]["answer_relevance"]["common_n"] == 2      # relevance у обоих есть
    assert tb["faith_qwen"]["none"] == 1
    assert tb["faith_qwen"]["abstained"] == 1
    assert tb["faith_qwen"]["failure"] == 0                          # None = воздержание, не сбой


def test_trusted_baseline_by_route():
    d = {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}
    cross = [_full("t1", d, d, route="factual"), _full("t2", d, d, route="multihop")]
    tb = trusted_baseline(cross)
    assert set(tb["by_route"]) == {"factual", "multihop"}
    assert tb["by_route"]["factual"]["faithfulness"]["common_n"] == 1


def test_calibration_split_separates_refusal_and_nonrefusal():
    g = {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}
    cross = [
        _full("r1", {"answer_relevance": 0.0, "faithfulness": 1.0, "context_precision": 1.0},
              {"answer_relevance": 1.0, "faithfulness": 1.0, "context_precision": 1.0}, abst=True),
        _full("n1", {"answer_relevance": 1.0, "faithfulness": 1.0, "context_precision": 1.0},
              {"answer_relevance": 0.0, "faithfulness": 1.0, "context_precision": 1.0}),
    ]
    gold = {"r1": g, "n1": g}
    split = calibration_split(cross, gold)
    assert split["refusal"]["n"] == 1 and split["non_refusal"]["n"] == 1
    # на отказе DeepSeek занижал relevance -> его MAE хуже; на не-отказе — наоборот Qwen хуже
    assert split["refusal"]["answer_relevance"]["deepseek_mae"] == 1.0
    assert split["non_refusal"]["answer_relevance"]["qwen_mae"] == 1.0


def test_calibration_split_empty_nonrefusal_is_honest():
    """Пустой non-refusal gold -> вердикт честно говорит «глобальность не проверить»."""
    g = {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}
    cross = [_full("r1", g, g, abst=True)]  # только отказ
    split = calibration_split(cross, {"r1": g})
    assert split["non_refusal"]["n"] == 0
    assert "не проверить" in split["verdict"].lower() or "глобальность" in split["verdict"].lower()


def test_calibration_split_global_verdict_when_qwen_better():
    """Qwen ближе к человеку и на не-отказах -> глобальное доверие."""
    g = {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0}
    # не-отказ: Qwen точен, DeepSeek мажет
    cross = [_full("n1", {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0},
                   {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0})]
    split = calibration_split(cross, {"n1": g})
    assert "ГЛОБАЛЬНО" in split["verdict"]
