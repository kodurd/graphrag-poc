"""Гвард измерений — на фейковых ретриверах/коннекторе, без сети и графа."""

from __future__ import annotations

import pytest

from eval.measure_guard import (
    config_fingerprint,
    diff_config_axes,
    diff_graph_fingerprints,
    graph_fingerprint,
    preflight_neighbor_check,
)


class FakeConn:
    """Отдаёт фиксированный count на каждый запрос по порядку ключей fingerprint."""

    def __init__(self, values: dict[str, int]):
        # порядок соответствует последовательности one(...) в graph_fingerprint
        self._seq = [
            values["nodes"], values["chunks_total"], values["chunks_page"],
            values["kip_pages"], values["kip_pages_sectioned"],
        ]
        self._i = 0

    def run(self, _q: str):
        v = self._seq[self._i]
        self._i += 1
        return [{"x": v}]


class FakeRetriever:
    """retrieve(q) -> заранее заданный список кандидатов по вопросу."""

    def __init__(self, by_q: dict[str, list[str]]):
        self._by_q = by_q

    def retrieve(self, q: str):
        return {"candidates": [{"id": i} for i in self._by_q.get(q, [])]}


def test_graph_fingerprint_reads_all_counts():
    vals = dict(nodes=100, chunks_total=50, chunks_page=40, kip_pages=490, kip_pages_sectioned=490)
    assert graph_fingerprint(FakeConn(vals)) == vals


def test_preflight_passes_when_neighbors_add_chunks():
    with_ = FakeRetriever({"q1": ["a", "b", "c"]})       # +1 сосед
    without = FakeRetriever({"q1": ["a", "b"]})
    assert preflight_neighbor_check(with_, without, ["q1"]) == 1


def test_preflight_fails_when_flag_is_noop():
    # Ровно баг −10.7пп: флаг «включён», но кандидаты не изменились.
    same = FakeRetriever({"q1": ["a", "b"], "q2": ["c"]})
    with pytest.raises(AssertionError, match="PREFLIGHT"):
        preflight_neighbor_check(same, same, ["q1", "q2"])


def test_diff_graph_fingerprints_detects_rebuild():
    before = dict(nodes=100, chunks_total=50, chunks_page=40, kip_pages=490, kip_pages_sectioned=2)
    after = dict(nodes=100, chunks_total=8000, chunks_page=7990, kip_pages=490, kip_pages_sectioned=490)
    diffs = diff_graph_fingerprints(before, after)
    assert any("chunks_total" in d for d in diffs)
    assert any("kip_pages_sectioned" in d for d in diffs)


def test_diff_graph_fingerprints_identical_is_clean():
    fp = dict(nodes=1, chunks_total=1, chunks_page=1, kip_pages=1, kip_pages_sectioned=1)
    assert diff_graph_fingerprints(fp, dict(fp)) == []


def test_diff_graph_fingerprints_missing_warns():
    assert diff_graph_fingerprints(None, {}) != []          # старый прогон -> предупреждение


def test_diff_config_axes_isolates_single_axis():
    before = config_fingerprint(answer_mode="default", kip_reserve=2, kip_neighbors=0, top_k=8, rerank_top_k=5)
    after = config_fingerprint(answer_mode="default", kip_reserve=2, kip_neighbors=1, top_k=8, rerank_top_k=5)
    assert diff_config_axes(before, after) == ["kip_neighbors"]


def test_diff_config_axes_empty_means_noop():
    fp = config_fingerprint(answer_mode="default", kip_reserve=2, kip_neighbors=0, top_k=8, rerank_top_k=5)
    assert diff_config_axes(fp, dict(fp)) == []              # мерили одно и то же
