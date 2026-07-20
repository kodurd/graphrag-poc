"""Общие фикстуры. Интеграционные тесты требуют поднятый Neo4j."""

from __future__ import annotations

import pytest

from graphrag.config import load_settings
from graphrag.graph.connection import Neo4jConnection


@pytest.fixture
def neo4j_conn():
    """Подключение к локальному Neo4j; пропуск теста, если он недоступен.

    Чистит БД до и после теста (локальная dev-БД PoC).
    """
    settings = load_settings()
    conn = Neo4jConnection(settings.neo4j)
    if not conn.verify_connectivity():
        conn.close()
        pytest.skip("Neo4j недоступен — запустите `docker compose up -d`")
    _reset(conn)
    try:
        yield conn
    finally:
        _reset(conn)
        conn.close()


def _reset(conn: Neo4jConnection) -> None:
    """Чистый лист: данные + схема (constraints, затем индексы).

    Индексы сбрасываются, потому что второй vector-индекс на той же паре
    (label, property) создать нельзя — иначе тесты цепляются друг за друга.
    """
    conn.run("MATCH (n) DETACH DELETE n")
    for row in conn.run("SHOW CONSTRAINTS YIELD name RETURN name"):
        conn.run(f"DROP CONSTRAINT {row['name']} IF EXISTS")
    for row in conn.run("SHOW INDEXES YIELD name RETURN name"):
        conn.run(f"DROP INDEX {row['name']} IF EXISTS")
