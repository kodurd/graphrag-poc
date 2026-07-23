"""U1: калибровка faithfulness-судьи — чистые метрики согласия с ручным gold."""

from __future__ import annotations

import pytest

from eval.faith_calib import judge_agreement, run_gold_judge


def _p(human, judge, abstained=False):
    return {"human": human, "judge": judge, "abstained": abstained}


def test_perfect_agreement():
    pairs = [_p(1.0, 1.0), _p(0.0, 0.0), _p(0.5, 0.5)]
    a = judge_agreement(pairs)
    assert a["mae"] == pytest.approx(0.0)
    assert a["bucket_agreement"] == pytest.approx(1.0)
    assert a["directional_residual"] == pytest.approx(0.0)
    assert a["n_scored"] == 3


def test_systematic_underscoring_gives_negative_residual():
    # судья везде занижает → направленный остаток < 0 (сигнал смещения, не шума)
    pairs = [_p(1.0, 0.0), _p(1.0, 0.0), _p(1.0, 0.0)]
    a = judge_agreement(pairs)
    assert a["directional_residual"] == pytest.approx(-1.0)
    assert a["mae"] == pytest.approx(1.0)


def test_symmetric_noise_has_near_zero_residual_but_positive_mae():
    # расхождение в обе стороны → остаток ≈ 0, но mae > 0 (шумовая сигнатура)
    pairs = [_p(0.5, 1.0), _p(0.5, 0.0)]
    a = judge_agreement(pairs)
    assert a["directional_residual"] == pytest.approx(0.0)
    assert a["mae"] == pytest.approx(0.5)


def test_true_negative_counts_as_agreement():
    # реально-неверный ответ, судья верно поставил 0 → согласие по обе стороны
    pairs = [_p(0.0, 0.0), _p(1.0, 1.0)]
    a = judge_agreement(pairs)
    assert a["bucket_agreement"] == pytest.approx(1.0)


def test_partial_bucket_participates():
    pairs = [_p(0.5, 0.45)]  # оба -> бакет 0.5
    a = judge_agreement(pairs)
    assert a["bucket_agreement"] == pytest.approx(1.0)
    assert a["mae"] == pytest.approx(0.05)


def test_bucket_mismatch_detected():
    pairs = [_p(1.0, 0.4)]  # human -> 1.0, judge 0.4 -> бакет 0.5
    a = judge_agreement(pairs)
    assert a["bucket_agreement"] == pytest.approx(0.0)


def test_abstained_excluded_and_counted():
    pairs = [_p(1.0, 1.0), _p(0.0, None, abstained=True)]
    a = judge_agreement(pairs)
    assert a["n_scored"] == 1 and a["n_abstained"] == 1
    assert a["mae"] == pytest.approx(0.0)  # abstained не влияет на mae


def test_empty_is_safe():
    a = judge_agreement([])
    assert a["n_scored"] == 0
    assert a["mae"] is None and a["directional_residual"] is None


def test_run_gold_judge_builds_pairs_with_inline_context():
    calls = []

    def fake_judge(answer, context_texts):
        calls.append((answer, tuple(context_texts)))
        return (0.7, False)

    gold = [{"answer": "A", "context_text": "CTX", "human_faithfulness": 1.0}]
    pairs = run_gold_judge(gold, fake_judge)
    assert pairs == [{"human": 1.0, "judge": 0.7, "abstained": False}]
    assert calls == [("A", ("CTX",))]  # контекст инлайн, одним элементом


def test_run_gold_judge_propagates_abstention():
    gold = [{"answer": "A", "context_text": "CTX", "human_faithfulness": 0.0}]
    pairs = run_gold_judge(gold, lambda a, c: (None, True))
    assert pairs[0]["judge"] is None and pairs[0]["abstained"] is True


def test_seed_gold_is_balanced_with_inline_context():
    import json
    from pathlib import Path

    gold = json.loads(Path("eval/trial/faith_gold.json").read_text(encoding="utf-8"))
    items = gold["items"]
    labels = {it["human_faithfulness"] for it in items}
    assert {0.0, 0.5, 1.0} <= labels  # сбалансирован: есть true-negatives, partial и faithful
    for it in items:  # контекст инлайн + per-claim обоснование
        assert it["context_text"] and it["claims_note"]
