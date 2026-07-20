"""Entity resolution — нормализация/кластеризация (чисто) + слияние (Neo4j)."""

from __future__ import annotations

import pytest

from graphrag.embeddings.embedder import HashingEmbedder
from graphrag.graph.resolve import EntityResolver, cluster_key, find_clusters, normalize_name
from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.intermediate import edge, node


# --- нормализация / ключи ---

def test_normalize_drops_stopwords_and_order():
    assert normalize_name("Auth Service") == normalize_name("service auth") == "auth"


def test_cluster_key_prefers_canonical_dictionary():
    canon = {"сервис авторизации": "canon:auth", "ас авторизация": "canon:auth"}
    assert cluster_key("Сервис Авторизации", canon) == "canon:auth"
    assert cluster_key("АС Авторизация", canon) == "canon:auth"


# --- кластеризация ---

def test_find_clusters_canonical_unifies_cross_language():
    nodes = [
        {"id": "module:a", "name": "auth-service"},
        {"id": "module:b", "name": "сервис авторизации"},
        {"id": "module:c", "name": "АС Авторизация"},
        {"id": "module:d", "name": "billing"},
    ]
    canon = {
        "auth-service": "canon:auth",
        "сервис авторизации": "canon:auth",
        "ас авторизация": "canon:auth",
    }
    clusters = find_clusters(nodes, canonical=canon)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"module:a", "module:b", "module:c"}


def test_find_clusters_embedding_merges_similar_not_disjoint():
    nodes = [
        {"id": "1", "name": "kafka connect worker"},
        {"id": "2", "name": "kafka connect task"},   # пересечение токенов
        {"id": "3", "name": "weather dashboard"},    # непересекающийся
    ]
    clusters = find_clusters(nodes, embedder=HashingEmbedder(256), threshold=0.5)
    merged = {frozenset(c) for c in clusters}
    assert frozenset({"1", "2"}) in merged
    assert all("3" not in c for c in clusters)  # непохожий не слит


# --- интеграция: слияние в графе ---

@pytest.mark.integration
def test_resolver_merges_duplicates_preserving_edges(neo4j_conn):
    apply_schema(neo4j_conn)
    load_records(neo4j_conn, [
        node("Module", "module:a", {"name": "auth-service"}),
        node("Module", "module:b", {"name": "сервис авторизации"}),
        node("Module", "module:c", {"name": "АС Авторизация"}),
        node("Module", "module:billing", {"name": "billing"}),
        # рёбра на разные дубли — после слияния должны сойтись на один узел
        node("Task", "task:1", {"summary": "auth bug"}),
        edge("MENTIONS", "task:1", "module:b"),
        edge("DEPENDS_ON", "module:a", "module:billing"),
    ])

    canon = {
        "auth-service": "canon:auth",
        "сервис авторизации": "canon:auth",
        "ас авторизация": "canon:auth",
    }
    stats = EntityResolver(neo4j_conn, canonical=canon).resolve()
    assert stats["merged_nodes"] == 2  # трое -> один, слито двое

    # Остался один auth-узел (+ billing).
    auth_ct = neo4j_conn.run(
        "MATCH (m:Module) WHERE m.name IN ['auth-service','сервис авторизации','АС Авторизация'] "
        "RETURN count(m) AS c"
    )[0]["c"]
    assert auth_ct == 1

    # Рёбра сохранены и сошлись на выживший узел (обход не оборвался).
    mentions = neo4j_conn.run(
        "MATCH (:Task {id:'task:1'})-[:MENTIONS]->(m:Module) RETURN count(m) AS c"
    )[0]["c"]
    assert mentions == 1
    depends = neo4j_conn.run(
        "MATCH (m:Module)-[:DEPENDS_ON]->(:Module {id:'module:billing'}) RETURN count(*) AS c"
    )[0]["c"]
    assert depends == 1
