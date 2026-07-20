"""Инкрементальный ре-sync — диф (чистый) + применение (Neo4j)."""

from __future__ import annotations

import pytest

from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.incremental.sync import (
    IncrementalSync,
    build_manifest,
    content_hash,
    diff_manifests,
    record_key,
)
from graphrag.intermediate import edge, node


# --- чистый диф ---

def test_record_key_node_and_edge():
    assert record_key(node("Task", "task:1", {})) == "node:task:1"
    assert record_key(edge("MENTIONS", "a", "b")) == "edge:a|MENTIONS|b"


def test_content_hash_changes_with_props():
    a = node("Task", "task:1", {"summary": "x"})
    b = node("Task", "task:1", {"summary": "x"})
    c = node("Task", "task:1", {"summary": "y"})
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_diff_manifests_classifies():
    old = build_manifest([
        node("Task", "task:1", {"s": "a"}),
        node("Task", "task:2", {"s": "a"}),
        node("Task", "task:3", {"s": "a"}),
    ])
    new = build_manifest([
        node("Task", "task:1", {"s": "a"}),   # unchanged
        node("Task", "task:2", {"s": "b"}),   # changed
        node("Task", "task:4", {"s": "a"}),   # added ; task:3 removed
    ])
    d = diff_manifests(old, new)
    assert d["added"] == {"node:task:4"}
    assert d["changed"] == {"node:task:2"}
    assert d["removed"] == {"node:task:3"}
    assert d["unchanged"] == {"node:task:1"}


# --- интеграция: применение дифа ---

@pytest.mark.integration
def test_apply_touches_only_changed(neo4j_conn):
    apply_schema(neo4j_conn)
    initial = [
        node("Task", "task:1", {"summary": "a"}),
        node("Task", "task:2", {"summary": "b"}),
        node("Task", "task:3", {"summary": "c"}),
    ]
    load_records(neo4j_conn, initial)
    # чанк, висящий на task:3 — должен уйти вместе с узлом
    load_records(neo4j_conn, [
        node("Chunk", "chunk:task:3#0", {"text": "c"}),
        edge("PART_OF", "chunk:task:3#0", "task:3"),
    ])
    manifest = build_manifest(initial)

    new = [
        node("Task", "task:1", {"summary": "a"}),    # unchanged
        node("Task", "task:2", {"summary": "b2"}),   # changed
        node("Task", "task:4", {"summary": "d"}),    # added ; task:3 removed
    ]
    stats, _ = IncrementalSync(neo4j_conn).apply(new, manifest)

    assert stats == {"added": 1, "changed": 1, "removed": 1, "unchanged": 1, "touched": 3}

    # состояние графа
    t2 = neo4j_conn.run("MATCH (t:Task {id:'task:2'}) RETURN t.summary AS s")[0]["s"]
    assert t2 == "b2"  # изменённый обновлён
    assert neo4j_conn.run("MATCH (t:Task {id:'task:1'}) RETURN t.summary AS s")[0]["s"] == "a"
    assert neo4j_conn.run("MATCH (t:Task {id:'task:4'}) RETURN count(t) AS c")[0]["c"] == 1
    assert neo4j_conn.run("MATCH (t:Task {id:'task:3'}) RETURN count(t) AS c")[0]["c"] == 0
    # чанк удалённого узла тоже снят (нет висячих рёбер)
    assert neo4j_conn.run("MATCH (c:Chunk {id:'chunk:task:3#0'}) RETURN count(c) AS c")[0]["c"] == 0


@pytest.mark.integration
def test_apply_new_commit_adds_without_full_reindex(neo4j_conn):
    apply_schema(neo4j_conn)
    initial = [node("Commit", "commit:a", {"message": "first"})]
    load_records(neo4j_conn, initial)
    manifest = build_manifest(initial)

    new = initial + [
        node("Commit", "commit:b", {"message": "second"}),
        node("File", "file:x.java", {"path": "x.java"}),
        edge("TOUCHES", "commit:b", "file:x.java"),
    ]
    stats, _ = IncrementalSync(neo4j_conn).apply(new, manifest)
    assert stats["added"] == 3 and stats["changed"] == 0 and stats["unchanged"] == 1
    assert neo4j_conn.run(
        "MATCH (:Commit {id:'commit:b'})-[:TOUCHES]->(:File {id:'file:x.java'}) RETURN count(*) AS c"
    )[0]["c"] == 1
