"""Метрики (чистые) + golden set и retrieval-оценка (интеграция)."""

from __future__ import annotations

import pytest

from eval.metrics import (
    candidate_entity_id,
    edge_precision_recall,
    judge_answer_correctness,
    judge_answer_relevance,
    judge_context_precision,
    judge_context_recall,
    judge_faithfulness,
    precision_recall_f1,
    recall_at_k,
)
from graphrag.llm.base import LLMClient


def _scripted(payload: str) -> LLMClient:
    """LLM-судья, всегда отдающий заданную строку ответа."""
    class ScriptedLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return payload

    return ScriptedLLM("x")


def _exploding() -> LLMClient:
    class BadLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            raise RuntimeError("boom")

    return BadLLM("x")


# --- чистые метрики ---

def test_precision_recall_f1():
    m = precision_recall_f1(["a", "b", "x"], ["a", "b", "c"])
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_precision_recall_empty_expected():
    m = precision_recall_f1(["a"], [])
    assert m["recall"] == 0.0 and m["f1"] == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c", "d"], ["c", "z"], k=3) == pytest.approx(0.5)


def test_edge_precision_recall():
    pred = [("a", "DEPENDS_ON", "b"), ("a", "DEPENDS_ON", "c")]
    gold = [("a", "DEPENDS_ON", "b")]
    m = edge_precision_recall(pred, gold)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_candidate_entity_id():
    assert candidate_entity_id("chunk:task:KAFKA-1#0") == "task:KAFKA-1"
    assert candidate_entity_id("module:clients") == "module:clients"


def test_faith_prompt_scores_claims_not_hedge():
    """Промпт велит оценивать утверждения, а оговорка не обнуляет остальные."""
    from eval.metrics import _FAITH_PROMPT

    low = _FAITH_PROMPT.lower()
    assert "только сами утверждения" in low  # анти-anchoring: оценивать утверждения
    assert "не понижают балл" in low  # оговорка/отказ не штрафуется
    assert "abstained" in low and "null" in low  # воздержание -> null/abstained


def test_judge_faithfulness_parses_score():
    """Валидное число -> (score, abstained=False)."""
    score, abstained = judge_faithfulness(_scripted('{"faithfulness": 0.8}'), "ответ", ["контекст"])
    assert score == pytest.approx(0.8) and abstained is False
    # клампинг
    hi, _ = judge_faithfulness(_scripted('{"faithfulness": 1.9}'), "ответ", ["к"])
    assert hi == pytest.approx(1.0)


def test_judge_faithfulness_abstention_via_flag_or_null():
    """Воздержание -> (None, True); отличается от сбоя."""
    s1, a1 = judge_faithfulness(_scripted('{"faithfulness": null, "abstained": true}'), "о", ["к"])
    assert s1 is None and a1 is True
    s2, a2 = judge_faithfulness(_scripted('{"faithfulness": null}'), "о", ["к"])  # явный null
    assert s2 is None and a2 is True


def test_judge_faithfulness_failure_not_abstention():
    """Сбой судьи -> (None, False) — отличается от воздержания."""
    assert judge_faithfulness(_exploding(), "ответ", ["к"]) == (None, False)
    assert judge_faithfulness(_scripted("совсем не json"), "ответ", ["к"]) == (None, False)
    # валидный JSON без ключа faithfulness -> сбой, не воздержание
    assert judge_faithfulness(_scripted('{"что-то": 1}'), "ответ", ["к"]) == (None, False)


# --- контролируемые кейсы (детерминированный вход, не флаки-перепрогон) ---

def test_grounded_answer_with_hedge_not_zeroed():
    """Содержательный ответ с оговоркой не обнуляется (судья по скрипту 0.9)."""
    ANSWER = ("На основе контекста: методы реализуются через генератор, а не "
              "хардкодом. Конкретную реализацию по контексту описать невозможно.")
    score, abstained = judge_faithfulness(_scripted('{"faithfulness": 0.9}'), ANSWER, ["ctx"])
    assert score == pytest.approx(0.9) and abstained is False  # не 0, не воздержание


def test_hallucination_still_scored_low():
    """Вымысел, противоречащий контексту, по-прежнему низкий (судья по скрипту 0.0)."""
    score, abstained = judge_faithfulness(_scripted('{"faithfulness": 0.0}'), "выдуманный факт", ["ctx"])
    assert score == pytest.approx(0.0) and abstained is False  # низкий, но НЕ воздержание


# --- судьи reference-free и reference-required ---

def test_answer_relevance_parses_and_clamps():
    assert judge_answer_relevance(_scripted('{"answer_relevance": 0.7}'), "в?", "о") == pytest.approx(0.7)
    # Значение вне [0,1] клампится.
    assert judge_answer_relevance(_scripted('{"answer_relevance": 1.9}'), "в?", "о") == pytest.approx(1.0)


def test_context_precision_parses_and_clamps():
    llm = _scripted('{"context_precision": 0.5}')
    assert judge_context_precision(llm, "в?", ["a", "b"]) == pytest.approx(0.5)
    assert judge_context_precision(_scripted('{"context_precision": -3}'), "в?", ["a"]) == pytest.approx(0.0)


def test_answer_correctness_high_when_matching_reference():
    llm = _scripted('{"answer_correctness": 0.95}')
    assert judge_answer_correctness(llm, "в?", "ответ", "ответ") == pytest.approx(0.95)


def test_context_recall_high_when_reference_covered():
    llm = _scripted('{"context_recall": 1.0}')
    assert judge_context_recall(llm, "эталон", ["эталон целиком в контексте"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda llm: judge_answer_relevance(llm, "в?", "о"),
        lambda llm: judge_context_precision(llm, "в?", ["ctx"]),
        lambda llm: judge_answer_correctness(llm, "в?", "о", "эталон"),
        lambda llm: judge_context_recall(llm, "эталон", ["ctx"]),
    ],
)
def test_judges_return_none_on_failure(call):
    """Сбой судьи и мусор вместо JSON -> None (а не 0.0)."""
    assert call(_exploding()) is None
    assert call(_scripted("совсем не json")) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda llm: judge_answer_relevance(llm, "в?", "о"),
        lambda llm: judge_context_precision(llm, "в?", ["ctx"]),
        lambda llm: judge_answer_correctness(llm, "в?", "о", "эталон"),
        lambda llm: judge_context_recall(llm, "эталон", ["ctx"]),
    ],
)
def test_judges_return_none_on_wrong_key(call):
    """Валидный JSON без ожидаемого ключа -> None."""
    assert call(_scripted('{"что-то_другое": 0.5}')) is None


# --- интеграция: golden set из графа ---

@pytest.mark.integration
def test_golden_set_and_retrieval_eval(neo4j_conn):
    from graphrag.embeddings.embedder import HashingEmbedder
    from graphrag.embeddings.reranker import LexicalReranker
    from graphrag.graph.schema import apply_schema
    from graphrag.graph.skeleton import load_records
    from graphrag.index.vector import VectorIndexer, collect_text_nodes
    from graphrag.intermediate import edge, node
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.golden_set import build_from_graph, evaluate_retrieval

    apply_schema(neo4j_conn)
    load_records(neo4j_conn, [
        node("Module", "module:clients", {"name": "clients"}),
        node("Task", "task:1", {"key": "KAFKA-1", "summary": "clients reconnect loop under outage"}),
        node("Task", "task:2", {"key": "KAFKA-2", "summary": "clients timeout on send"}),
        edge("MENTIONS", "task:1", "module:clients"),
        edge("MENTIONS", "task:2", "module:clients"),
    ])
    emb = HashingEmbedder(dimension=128)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    golden = build_from_graph(neo4j_conn)
    assert any(g["kind"] == "module" for g in golden)
    module_q = next(g for g in golden if g["kind"] == "module")
    assert set(module_q["expected_ids"]) == {"task:1", "task:2"}

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker(), top_k=10, rerank_top_k=10)
    report = evaluate_retrieval(retr, golden)
    assert report["n"] >= 1
    assert report["recall"] > 0.0  # тикеты по модулю действительно находятся
