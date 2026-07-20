"""Сборка отчёта — чистые функции над синтетическими результатами."""

from __future__ import annotations

import pytest

from eval.quality_report import (
    abstention_stats,
    breakdown,
    breakdown_by_source,
    failures,
    render_report,
    source_type,
    summarize_metric,
)


def _rec(question, route, source_id, abstained=None, **metrics):
    r = {
        "question": question,
        "route": route,
        "source_id": source_id,
        "citations": ["https://issues/KAFKA-1"],
        "context_ids": ["chunk:task:1#0"],
        "metrics": metrics,
    }
    if abstained is not None:
        r["abstained"] = {"faithfulness": abstained}
    return r


RECORDS = [
    _rec("q1", "factual", "task:1", faithfulness=0.9, answer_relevance=0.8),
    _rec("q2", "multihop", "page:1", faithfulness=0.3, answer_relevance=0.2),
    _rec("q3", "factual", "task:2", faithfulness=0.6, answer_relevance=None),
]


# --- воздержания ---

def test_abstention_stats_distinguishes_abstention_failure_scored():
    recs = [
        _rec("a", "mixed", "task:1", abstained=True, faithfulness=None),   # воздержание
        _rec("b", "mixed", "task:2", abstained=False, faithfulness=None),  # сбой судьи
        _rec("c", "mixed", "task:3", abstained=False, faithfulness=0.7),   # оценено
        _rec("d", "mixed", "task:4", faithfulness=0.5),                    # нет поля -> оценено
    ]
    s = abstention_stats(recs)
    assert s["total"] == 4 and s["abstained"] == 1 and s["failed"] == 1 and s["scored"] == 2
    assert s["abstention_rate"] == pytest.approx(0.25)


def test_report_shows_abstention_rate_next_to_faithfulness():
    recs = [
        _rec("a", "mixed", "task:1", abstained=True, faithfulness=None),
        _rec("b", "mixed", "task:2", abstained=False, faithfulness=0.8),
    ]
    md = render_report({"records": recs, "counts": {}})
    assert "воздержаний 1/2" in md and "воздержание ≠ сбой" in md


def test_abstention_line_shown_even_when_all_abstained():
    # faithfulness не оценён (все воздержались) — строка воздержаний всё равно есть
    recs = [_rec("a", "mixed", "task:1", abstained=True, faithfulness=None)]
    md = render_report({"records": recs, "counts": {}})
    assert "воздержаний 1/1" in md


# --- агрегаты и распределения ---

def test_summarize_excludes_none_from_denominator():
    s = summarize_metric(RECORDS, "answer_relevance")
    # Оценены 2 из 3 (одна None) — среднее по двум, а не по трём.
    assert s["n"] == 2 and s["total"] == 3
    assert s["mean"] == pytest.approx(0.5)


def test_summarize_reports_distribution_not_just_mean():
    s = summarize_metric(RECORDS, "faithfulness")
    assert s["mean"] == pytest.approx(0.6)
    assert {"p10", "p50", "p90", "histogram"} <= set(s)
    assert sum(s["histogram"].values()) == 3


def test_metric_with_all_none_marked_not_scored():
    recs = [_rec("q", "factual", "task:1", faithfulness=None)]
    s = summarize_metric(recs, "faithfulness")
    assert s["scored"] is False and s["n"] == 0
    # И в отчёте это «не оценено», а не 0.
    assert "_не оценено_" in render_report({"records": recs, "counts": {}})


# --- разбивки ---

def test_breakdown_by_route():
    by_route = breakdown(RECORDS, "faithfulness", "route")
    assert by_route["factual"] == pytest.approx(0.75)  # (0.9 + 0.6) / 2
    assert by_route["multihop"] == pytest.approx(0.3)


def test_source_type_derived_from_id_prefix():
    assert source_type({"source_id": "task:KAFKA-1"}) == "Task"
    assert source_type({"source_id": "page:9"}) == "Page"
    assert source_type({"source_id": None}) == "unknown"


def test_breakdown_by_source_type():
    by_src = breakdown_by_source(RECORDS, "faithfulness")
    assert by_src["Task"] == pytest.approx(0.75)
    assert by_src["Page"] == pytest.approx(0.3)


# --- провалы ---

def test_failures_below_threshold_worst_first():
    bad = failures(RECORDS, "faithfulness", threshold=0.7)
    assert [r["question"] for r in bad] == ["q2", "q3"]  # 0.3 затем 0.6


def test_failures_ignores_none():
    bad = failures(RECORDS, "answer_relevance", threshold=1.0)
    assert [r["question"] for r in bad] == ["q2", "q1"]  # q3 (None) не попадает


# --- рендер ---

def test_report_has_caveat_metrics_and_failures():
    md = render_report({"records": RECORDS, "counts": {"questions": 3, "labeled": 0, "total": 3}})
    assert "Само-оценка" in md  # оговорка про смещение обязательна
    assert "Faithfulness" in md and "Примеры провалов" in md
    assert "по маршруту" in md and "по типу источника" in md


def test_retrieval_rendered_as_separate_section():
    md = render_report(
        {"records": RECORDS, "counts": {}},
        retrieval={"population": "graph-golden", "n": 41, "precision": 0.1, "recall": 0.2, "f1": 0.13},
    )
    assert "Retrieval P/R/F1 (отдельная популяция)" in md
    # Явно сказано, что цифры не сопоставимы с метриками качества выше.
    assert "не сопоставимы" in md
