"""Векторный индекс в Neo4j — интеграция (Hashing-эмбеддер, без ML-моделей)."""

from __future__ import annotations

import pytest

from graphrag.embeddings.embedder import HashingEmbedder
from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.index.vector import (
    VectorIndexer,
    collect_fix_commit_nodes,
    collect_text_nodes,
)
from graphrag.intermediate import node


class _FakeConn:
    """Заглушка Neo4jConnection: .run() отдаёт заранее заданные строки."""

    def __init__(self, rows):
        self._rows = rows

    def run(self, _query, **_params):
        return self._rows


def test_collect_fix_commit_nodes_filters_and_dedups():
    # Строки — то, что вернул бы Cypher MATCH (n:Commit)-[:MENTIONS]->(:Task).
    # Коммит без MENTIONS сюда не попадает по построению запроса (в графе);
    # здесь проверяем Python-логику: dedup по id + отброс пустого message.
    conn = _FakeConn([
        {"id": "commit:aaa", "text": "KAFKA-1: fix log4j bump", "uri": "https://c/aaa"},
        {"id": "commit:aaa", "text": "KAFKA-1: fix log4j bump", "uri": "https://c/aaa"},  # тот же коммит, 2-й тикет
        {"id": "commit:bbb", "text": "   ", "uri": "https://c/bbb"},  # пустой message → отброс
        {"id": "commit:ccc", "text": "KAFKA-9: bound elapsed time", "uri": None},  # uri может быть null
    ])
    out = collect_fix_commit_nodes(conn)

    ids = [row[0] for row in out]
    assert ids == ["commit:aaa", "commit:ccc"]  # dedup + отброс пустого
    assert out[0] == ("commit:aaa", "KAFKA-1: fix log4j bump", "https://c/aaa")
    assert out[1] == ("commit:ccc", "KAFKA-9: bound elapsed time", None)  # null uri сохранён


@pytest.mark.integration
def test_vector_search_finds_relevant_chunk(neo4j_conn):
    apply_schema(neo4j_conn)
    load_records(
        neo4j_conn,
        [
            node("Task", "task:1",
                 {"summary": "auth service login failure",
                  "description": "authentication NullPointerException on reconnect"},
                 {"uri": "https://issues/KAFKA-1"}),
            node("Task", "task:2",
                 {"summary": "weather widget layout",
                  "description": "UI spacing and colors for the dashboard"},
                 {"uri": "https://issues/KAFKA-2"}),
        ],
    )

    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb, index_name="test_chunk_emb")
    idx.ensure_index()

    nodes = collect_text_nodes(neo4j_conn)
    stats = idx.index_nodes(nodes)
    assert stats["chunks"] >= 2

    # Дождаться заполнения индекса (векторный индекс наполняется асинхронно).
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    res = idx.search("authentication auth service", top_k=2)
    assert res, "поиск ничего не вернул"
    assert res[0]["uri"] == "https://issues/KAFKA-1"  # релевантный чанк первым


@pytest.mark.integration
def test_chunk_carries_source_uri(neo4j_conn):
    apply_schema(neo4j_conn)
    load_records(
        neo4j_conn,
        [node("Page", "page:9", {"title": "KIP-1", "text": "содержимое страницы про kafka"},
              {"uri": "https://wiki/KIP-1"})],
    )
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb, index_name="test_chunk_emb2")
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))

    rows = neo4j_conn.run("MATCH (c:Chunk) RETURN c.uri AS uri, c.parent AS parent")
    assert rows
    assert all(r["uri"] == "https://wiki/KIP-1" for r in rows)
    assert all(r["parent"] == "page:9" for r in rows)
