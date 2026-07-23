"""U1: калибровка faithfulness-судьи — чистые метрики согласия с ручным gold."""

from __future__ import annotations

import pytest

from eval.faith_calib import (
    beats_baseline,
    diagnose,
    diagnose_run,
    judge_agreement,
    run_gold_judge,
    sample_variance,
)


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


def test_gold_has_hard_cases():
    import json
    from pathlib import Path

    gold = json.loads(Path("eval/trial/faith_gold.json").read_text(encoding="utf-8"))
    hard = [it for it in gold["items"] if it.get("hard")]
    assert len(hard) >= 4  # трудные случаи (длинные многоутверждённые how-to ответы) присутствуют
    for it in hard:
        assert len(it["answer"]) >= 200  # длинный ответ — не компактный ясный случай
        assert it["claims_note"].count(";") + it["claims_note"].count("\n") >= 1  # ≥2 утверждения разобраны
        assert it["context_text"] and it["claims_note"]
    hard_labels = {it["human_faithfulness"] for it in hard}
    assert len(hard_labels) >= 2  # трудные охватывают >1 направления (ловят ошибки в обе стороны)


# --- U2: диагностика шум vs смещение ---

def test_sample_variance_identical_is_zero():
    assert sample_variance([0.7, 0.7, 0.7]) == pytest.approx(0.0)


def test_sample_variance_bimodal_is_high():
    assert sample_variance([0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.25)  # разброс = шум


def test_sample_variance_ignores_none_and_short():
    assert sample_variance([0.5, None]) == pytest.approx(0.0)  # один скор -> нет дисперсии
    assert sample_variance([]) == pytest.approx(0.0)


def test_diagnose_noise_high_var_low_residual():
    assert diagnose(variance=0.2, residual=0.02)["verdict"] == "noise"


def test_diagnose_bias_low_var_high_residual():
    # судья стабилен (низкая дисперсия), но систематически занижает (большой |остаток|)
    assert diagnose(variance=0.01, residual=-0.6)["verdict"] == "bias"


def test_diagnose_mixed_both_high():
    assert diagnose(variance=0.2, residual=-0.6)["verdict"] == "mixed"


def test_diagnose_ok_both_low():
    assert diagnose(variance=0.01, residual=0.02)["verdict"] == "ok"


def test_diagnose_run_bias_signature():
    # temp=0 судья стабильно занижает faithful (residual<0), сэмплы почти без разброса → bias
    baseline = [_p(1.0, 0.0), _p(1.0, 0.0), _p(1.0, 0.0)]
    sampled = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]
    assert diagnose_run(baseline, sampled)["verdict"] == "bias"


def test_beats_baseline_handles_perfect_zero_baseline():
    assert beats_baseline(0.029, 0.0) is False   # 0.0 baseline валиден, не «отсутствует»
    assert beats_baseline(0.05, 0.1) is True
    assert beats_baseline(0.1, 0.1) is False      # строго лучше, не равно
    assert beats_baseline(None, 0.1) is False and beats_baseline(0.1, None) is False


def test_diagnose_run_noise_signature():
    # temp=0 в среднем согласуется (residual≈0), но сэмплы двухполюсны → noise
    baseline = [_p(0.5, 1.0), _p(0.5, 0.0)]
    sampled = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    assert diagnose_run(baseline, sampled)["verdict"] == "noise"
