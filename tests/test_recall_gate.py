"""U1: recall-гейт — есть ли source_id вопроса в пуле кандидатов ДО реранка."""

from __future__ import annotations

import pytest

from eval.recall_gate import evaluate_recall_gate, pool_entity_ids, source_in_pool


class _FakePool:
    """Стаб `_candidate_pool`: отдаёт заранее заданный (route, pool) по вопросу."""

    def __init__(self, by_question: dict):
        self._by = by_question

    def _candidate_pool(self, query: str):
        return self._by[query]


# --- чистые функции ---

def test_pool_entity_ids_reduces_chunks():
    pool = [{"id": "chunk:task:KAFKA-1#0"}, {"id": "module:connect"}]
    assert pool_entity_ids(pool) == ["task:KAFKA-1", "module:connect"]


def test_source_in_pool_hit_via_chunk():
    pool = [{"id": "chunk:task:KAFKA-1#0"}, {"id": "chunk:task:KAFKA-2#1"}]
    assert source_in_pool(pool, "task:KAFKA-1") == 1.0


def test_source_in_pool_hit_via_page():
    # candidate_entity_id сводит chunk:page:9#2 -> page:9
    assert source_in_pool([{"id": "chunk:page:9#2"}], "page:9") == 1.0


def test_source_in_pool_miss():
    assert source_in_pool([{"id": "chunk:task:KAFKA-9#0"}], "task:KAFKA-1") == 0.0


def test_source_in_pool_empty():
    assert source_in_pool([], "task:KAFKA-1") == 0.0


# --- агрегат со сплитом воздержаний ---

def test_evaluate_recall_gate_splits_abstained_answered():
    retr = _FakePool({
        "q_hit_abs": ("mixed", [{"id": "chunk:task:A#0"}]),     # воздержание, hit
        "q_miss_abs": ("mixed", [{"id": "chunk:task:Z#0"}]),    # воздержание, miss
        "q_hit_ans": ("factual", [{"id": "chunk:task:C#0"}]),   # ответ, hit
    })
    questions = [
        {"question": "q_hit_abs", "source_id": "task:A"},
        {"question": "q_miss_abs", "source_id": "task:B"},
        {"question": "q_hit_ans", "source_id": "task:C"},
    ]
    out = evaluate_recall_gate(retr, questions, abstained={"q_hit_abs", "q_miss_abs"})

    assert out["n_abstained"] == 2 and out["n_answered"] == 1
    assert out["hit_rate_abstained"] == pytest.approx(0.5)   # 1 из 2 воздержаний в пуле
    assert out["hit_rate_answered"] == pytest.approx(1.0)
    assert out["hit_rate_all"] == pytest.approx(2 / 3)

    rec = {r["question"]: r for r in out["records"]}
    assert rec["q_hit_abs"]["hit"] == 1.0 and rec["q_hit_abs"]["abstained"] is True
    assert rec["q_miss_abs"]["hit"] == 0.0
    assert rec["q_hit_ans"]["route"] == "factual" and rec["q_hit_ans"]["pool_size"] == 1
    assert rec["q_hit_ans"]["abstained"] is False
