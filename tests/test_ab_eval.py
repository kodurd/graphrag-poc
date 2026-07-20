"""U2: парный A/B lexical vs cross-encoder — чистые функции дельт/агрегатов/вердикта."""

from __future__ import annotations

import pytest

from eval.ab_eval import (
    changed_ranking,
    falsification_verdict,
    is_answered,
    paired_deltas,
    per_version_summary,
    render_ab_report,
    render_changed_ranking,
    run_ab_eval,
    version_abstention,
)


def _rrec(context_ids, answer="ответ"):
    return {"context_ids": context_ids, "answer": answer,
            "abstained": {"faithfulness": False}, "metrics": {}}


def _rec(*, abstained=False, **metrics):
    return {"abstained": {"faithfulness": abstained}, "metrics": metrics}


def _pair(question, lex, ce):
    return {"question": question, "lexical": lex, "cross_encoder": ce}


# --- is_answered ---

def test_is_answered_reads_abstention_flag():
    assert is_answered(_rec(abstained=False)) is True
    assert is_answered(_rec(abstained=True)) is False


# --- дельты только на совместно-отвечённых ---

def test_paired_deltas_only_jointly_answered():
    pairs = [
        _pair("both", _rec(answer_correctness=0.6), _rec(answer_correctness=0.9)),   # обе -> дельта
        _pair("lex_abst", _rec(abstained=True), _rec(answer_correctness=0.8)),       # lexical воздержался
        _pair("ce_none", _rec(answer_correctness=0.5), _rec(answer_correctness=None)),  # метрика None
    ]
    d = paired_deltas(pairs, "answer_correctness")
    assert [x["question"] for x in d] == ["both"]
    assert d[0]["delta"] == pytest.approx(0.3)


# --- пер-версийный агрегат по собственному множеству отвечённых ---

def test_per_version_summary_uses_own_answered_set():
    pairs = [
        _pair("q1", _rec(abstained=True), _rec(context_precision=0.8)),   # ответил только CE
        _pair("q2", _rec(context_precision=0.4), _rec(context_precision=0.6)),
    ]
    lex = per_version_summary(pairs, "lexical", "context_precision")
    ce = per_version_summary(pairs, "cross_encoder", "context_precision")
    assert lex["n"] == 1 and lex["mean"] == pytest.approx(0.4)           # q1 исключён (воздержался)
    assert ce["n"] == 2 and ce["mean"] == pytest.approx(0.7)             # q1 входит в CE


# --- воздержания по версиям ---

def test_version_abstention_rate():
    pairs = [
        _pair("q1", _rec(abstained=True), _rec()),
        _pair("q2", _rec(abstained=True), _rec()),
        _pair("q3", _rec(), _rec(abstained=True)),
    ]
    assert version_abstention(pairs, "lexical")["rate"] == pytest.approx(2 / 3)
    assert version_abstention(pairs, "cross_encoder")["rate"] == pytest.approx(1 / 3)


# --- фальсификационный вердикт ---

def test_verdict_supported_when_positive_on_at_least_K():
    deltas = [{"delta": 0.2}, {"delta": 0.1}, {"delta": -0.05}]
    assert falsification_verdict(deltas, K=2, abst_lex=0.5, abst_ce=0.4) == "supported"


def test_verdict_not_supported_on_zero_delta():
    deltas = [{"delta": 0.0}, {"delta": 0.0}]
    assert falsification_verdict(deltas, K=1, abst_lex=0.5, abst_ce=0.5) == "not_supported"


def test_verdict_directional_when_joint_below_K():
    deltas = [{"delta": 0.9}]
    assert falsification_verdict(deltas, K=3, abst_lex=0.5, abst_ce=0.5) == "directional_only"


def test_verdict_guardrail_ce_abstention_higher():
    deltas = [{"delta": 0.9}, {"delta": 0.9}]  # дельта отличная, но...
    assert falsification_verdict(deltas, K=1, abst_lex=0.4, abst_ce=0.6) == "not_supported"


# --- раннер: две записи на вопрос через инъекцию evaluate_fn ---

def test_run_ab_eval_two_records_per_question():
    calls = []

    def fake_eval(retr, llm, question, *, reference=None, source_id=None):
        calls.append(retr)
        return _rec(answer_correctness=0.5)

    pairs = run_ab_eval(
        "RETR_LEX", "RETR_CE", llm=None,
        questions=[{"question": "q1", "source_id": "task:A"}],
        labeled=[{"question": "q2", "reference": "эталон", "source_id": "task:B"}],
        evaluate_fn=fake_eval,
    )
    assert len(pairs) == 2
    assert set(pairs[0]) == {"question", "lexical", "cross_encoder"}
    assert calls == ["RETR_LEX", "RETR_CE", "RETR_LEX", "RETR_CE"]  # обе версии на каждый вопрос


# --- рендер отчёта (R6) ---

def test_render_ab_report_shows_versions_abstention_verdict():
    pairs = [
        _pair("q1", _rec(context_precision=0.4), _rec(context_precision=0.7)),
        _pair("q2", _rec(abstained=True), _rec(context_precision=0.6)),
    ]
    md = render_ab_report(pairs, ["context_precision"], K=1)
    assert "Воздержания: lexical 1/2" in md and "cross-encoder 0/2" in md
    assert "lexical: n=1" in md and "cross-encoder: n=2" in md  # пер-версийные множества
    assert "вердикт: **supported**" in md  # дельта +0.3 на 1 совместном, K=1
    assert "+0.30" in md  # таблица дельт


# --- U3: сверка изменившегося ранжирования ---

def test_changed_ranking_selects_only_differing():
    pairs = [
        _pair("same", _rrec(["a", "b"]), _rrec(["a", "b"])),   # идентичны -> исключён
        _pair("diff", _rrec(["a", "b"]), _rrec(["b", "a"])),   # порядок иной -> включён
    ]
    assert [p["question"] for p in changed_ranking(pairs)] == ["diff"]


def test_render_changed_ranking_contains_both_sides():
    pairs = [_pair("q", _rrec(["a"], "лекс-ответ"), _rrec(["b"], "це-ответ"))]
    md = render_changed_ranking(pairs)
    assert "лекс-ответ" in md and "це-ответ" in md
    assert "['a']" in md and "['b']" in md
    assert "смещение судьи" in md  # caveat на месте


def test_render_changed_ranking_empty_case():
    pairs = [_pair("same", _rrec(["a"]), _rrec(["a"]))]
    assert "Изменений ранжирования нет." in render_changed_ranking(pairs)
