"""Чистое ядро замера связи тикет→фикс: classify + aggregate."""

from __future__ import annotations

from eval.fix_linkage import aggregate, classify_linkage


# --- classify_linkage ---

def test_classify_three_classes():
    assert classify_linkage(has_fix=False, fix_reachable=False) == "no_fix"
    assert classify_linkage(has_fix=False, fix_reachable=True) == "no_fix"   # нет фикса — reachable неважен
    assert classify_linkage(has_fix=True, fix_reachable=True) == "already_retrieved"
    assert classify_linkage(has_fix=True, fix_reachable=False) == "surfaceable"


# --- aggregate (четыре вердикта) ---

def test_aggregate_surfacing():
    v = aggregate(["surfaceable"] * 7 + ["no_fix"] * 3)
    assert v["verdict"] == "surfacing" and v["frac"]["surfaceable"] == 0.7


def test_aggregate_ingest():
    v = aggregate(["no_fix"] * 6 + ["surfaceable"] * 4)
    assert v["verdict"] == "ingest"


def test_aggregate_content_problem():
    v = aggregate(["already_retrieved"] * 7 + ["surfaceable"] * 3)
    assert v["verdict"] == "content_problem"


def test_aggregate_mixed_when_no_majority():
    v = aggregate(["surfaceable"] * 4 + ["already_retrieved"] * 4 + ["no_fix"] * 2)
    assert v["verdict"] == "mixed"


def test_aggregate_empty():
    v = aggregate([])
    assert v["verdict"] == "mixed" and v["n"] == 0
    assert v["frac"]["surfaceable"] is None  # falsy-zero: пустой ≠ 0-доля
