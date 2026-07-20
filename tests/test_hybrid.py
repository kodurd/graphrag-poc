"""BM25 (чистый) + гибридный retrieval (интеграция)."""

from __future__ import annotations

import pytest

from graphrag.embeddings.embedder import HashingEmbedder
from graphrag.embeddings.reranker import LexicalReranker
from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.index.bm25 import BM25Index
from graphrag.index.vector import VectorIndexer, collect_text_nodes
from graphrag.intermediate import edge, node
from graphrag.retrieval.hybrid import HybridRetriever, filter_by_threshold
from graphrag.retrieval.router import FACTUAL, MULTIHOP


# --- порог релевантности (чистая функция) ---

def test_threshold_disabled_keeps_all():
    items = [{"id": "a", "source": "vector", "rerank_score": 0.1}]
    assert filter_by_threshold(items, 0.0) == items  # 0 = отключён
    assert filter_by_threshold(items, -1) == items


def test_threshold_drops_low_vector_candidates():
    items = [
        {"id": "hi", "source": "vector", "rerank_score": 0.8},
        {"id": "lo", "source": "bm25", "rerank_score": 0.2},
    ]
    kept = filter_by_threshold(items, 0.5)
    assert [it["id"] for it in kept] == ["hi"]  # низкий отброшен


def test_threshold_all_below_yields_empty():
    items = [{"id": "a", "source": "vector", "rerank_score": 0.1}]
    assert filter_by_threshold(items, 0.5) == []  # -> пусто -> честное «не знаю»


def test_threshold_exempts_graph_candidates():
    items = [
        {"id": "module:connect", "source": "graph", "rerank_score": 0.05},
        {"id": "chunk:lo", "source": "bm25", "rerank_score": 0.05},
    ]
    kept = filter_by_threshold(items, 0.5)
    assert [it["id"] for it in kept] == ["module:connect"]  # граф остаётся


# --- наблюдаемость source ---

def test_source_field_present_and_first_writer_wins():
    # источник наблюдаем; при дубле вектор раньше bm25 -> помечен vector
    items = [
        {"id": "x", "source": "vector", "rerank_score": 0.9},
        {"id": "y", "source": "graph", "rerank_score": 0.4},
    ]
    kept = filter_by_threshold(items, 0.0)
    assert {it["source"] for it in kept} == {"vector", "graph"}


# --- BM25 (чистый) ---

def test_bm25_ranks_overlap_higher():
    idx = BM25Index([
        {"id": "1", "text": "weather dashboard layout colors", "uri": "u1"},
        {"id": "2", "text": "kafka network client reconnect broker", "uri": "u2"},
    ])
    res = idx.search("network client broker", top_k=2)
    assert res and res[0]["id"] == "2"


def test_bm25_empty_corpus():
    assert BM25Index([]).search("что угодно") == []


# --- гибрид (интеграция) ---

def _seed(conn):
    apply_schema(conn)
    load_records(conn, [
        node("Module", "module:clients", {"name": "clients"}),
        node("Module", "module:connect", {"name": "connect"}),
        node("Module", "module:streams", {"name": "streams"}),
        edge("DEPENDS_ON", "module:connect", "module:clients"),
        edge("DEPENDS_ON", "module:streams", "module:clients"),
        node("Task", "task:1",
             {"summary": "NetworkClient reconnect", "description": "clients module broker handling",
              "uri": "https://issues/KAFKA-1"}),
        edge("MENTIONS", "task:1", "module:clients"),
    ])


@pytest.mark.integration
def test_multihop_surfaces_graph_only_modules(neo4j_conn):
    """Multi-hop через граф находит connect/streams, которых нет у вектора."""
    _seed(neo4j_conn)
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker(), max_hops=2)

    result = retr.retrieve("что зависит от clients")
    assert result["route"] == MULTIHOP
    ids = {c["id"] for c in result["candidates"]}
    assert "module:connect" in ids and "module:streams" in ids

    # вектор-only тот же запрос модули не вернёт (у них нет чанков)
    vec_ids = {r["id"] for r in idx.search("что зависит от clients", top_k=8)}
    assert "module:connect" not in vec_ids and "module:streams" not in vec_ids


@pytest.mark.integration
def test_factual_uses_vector_and_bm25(neo4j_conn):
    _seed(neo4j_conn)
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker())

    result = retr.retrieve("что такое NetworkClient")
    assert result["route"] == FACTUAL
    assert result["candidates"], "факт-запрос должен вернуть чанки"
    assert any("networkclient" in c["text"].lower() for c in result["candidates"])
