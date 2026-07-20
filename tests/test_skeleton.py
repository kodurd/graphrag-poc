"""Скелет графа — чистая группировка (без Neo4j) + интеграция (с Neo4j)."""

from __future__ import annotations

import pytest

from graphrag.graph.schema import apply_schema, label_of
from graphrag.graph.skeleton import load_records, plan_writes
from graphrag.intermediate import edge, node

FIXTURE = [
    node("Commit", "commit:abc", {"sha": "abc", "message": "KAFKA-101 fix"}, {"source": "git"}),
    node("Task", "task:KAFKA-101", {"key": "KAFKA-101", "summary": "fix"},
         {"source": "jira", "uri": "https://issues/KAFKA-101"}),
    node("Module", "module:clients", {"name": "clients"}),
    edge("MENTIONS", "commit:abc", "task:KAFKA-101"),
    edge("DUPLICATES", "task:KAFKA-101", "task:KAFKA-50"),  # KAFKA-50 — заглушка
]


# --- чистая группировка ---

def test_label_of_by_prefix():
    assert label_of("task:KAFKA-1") == "Task"
    assert label_of("commit:abc") == "Commit"
    assert label_of("unknown:x") == "Entity"


def test_plan_writes_groups_nodes_and_edges():
    nodes_by_label, edges_by_key = plan_writes(FIXTURE)
    assert set(nodes_by_label) == {"Commit", "Task", "Module"}
    assert ("Commit", "MENTIONS", "Task") in edges_by_key
    assert ("Task", "DUPLICATES", "Task") in edges_by_key


def test_plan_writes_folds_uri_from_source():
    nodes_by_label, _ = plan_writes(FIXTURE)
    task = nodes_by_label["Task"][0]
    assert task["uri"] == "https://issues/KAFKA-101"
    assert task["id"] == "task:KAFKA-101"


# --- интеграция с Neo4j ---

@pytest.mark.integration
def test_load_creates_nodes_and_edges(neo4j_conn):
    apply_schema(neo4j_conn)
    stats = load_records(neo4j_conn, FIXTURE)
    assert stats["nodes"] == 3

    counts = neo4j_conn.run(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label"
    )
    by_label = {row["label"]: row["c"] for row in counts}
    assert by_label["Commit"] == 1
    assert by_label["Module"] == 1
    # Task: KAFKA-101 (реальный) + KAFKA-50 (заглушка от DUPLICATES) = 2
    assert by_label["Task"] == 2

    rels = neo4j_conn.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c")
    rel_types = {row["t"]: row["c"] for row in rels}
    assert rel_types["MENTIONS"] == 1
    assert rel_types["DUPLICATES"] == 1


@pytest.mark.integration
def test_load_is_idempotent(neo4j_conn):
    apply_schema(neo4j_conn)
    load_records(neo4j_conn, FIXTURE)
    load_records(neo4j_conn, FIXTURE)  # повтор не должен плодить дубли
    total = neo4j_conn.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    assert total == 4  # Commit + 2 Task + Module
    rels = neo4j_conn.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    assert rels == 2


@pytest.mark.integration
def test_dangling_endpoint_becomes_stub(neo4j_conn):
    """Ребро на несуществующий узел создаёт заглушку с корректной меткой."""
    apply_schema(neo4j_conn)
    load_records(neo4j_conn, FIXTURE)
    stub = neo4j_conn.run("MATCH (t:Task {id: 'task:KAFKA-50'}) RETURN t.id AS id")
    assert stub and stub[0]["id"] == "task:KAFKA-50"
