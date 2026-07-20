"""Лог -> impact. Извлечение (чистое) + сквозной обход (Neo4j)."""

from __future__ import annotations

import pytest

from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.intermediate import edge, node
from graphrag.llm.base import LLMClient
from graphrag.retrieval.impact import ImpactAnalyzer, extract_entities

SAMPLE_LOG = """
2024-03-05 12:00:01 ERROR [worker-1] task failed
java.lang.IllegalStateException: broker unavailable
\tat org.apache.kafka.clients.NetworkClient.poll(NetworkClient.java:200)
\tat org.apache.kafka.clients.producer.KafkaProducer.send(KafkaProducer.java:1000)
"""


# --- извлечение сущностей (без LLM) ---

def test_extract_exceptions_modules_classes():
    ent = extract_entities(SAMPLE_LOG)
    assert "IllegalStateException" in ent["exceptions"]
    assert "clients" in ent["modules"]
    assert any("NetworkClient" in c for c in ent["classes"])


def test_extract_llm_adds_rare_entities():
    class ScriptedLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"modules": ["streams"], "services": [], "exceptions": ["TimeoutException"]}'

    ent = extract_entities(SAMPLE_LOG, ScriptedLLM("x"))
    assert "streams" in ent["modules"]  # добавлено LLM
    assert "clients" in ent["modules"]  # регексное сохранено
    assert "TimeoutException" in ent["exceptions"]


def test_extract_empty_log():
    ent = extract_entities("")
    assert ent == {"exceptions": [], "classes": [], "modules": []}


def test_extract_ignores_thread_names():
    """Имена потоков в [...] — не модули (worker-1, main, task-connect-sink-0)."""
    log = (
        "2024-03-06 12:00:00,000 ERROR [worker-1] failed (com.example.app.Widget)\n"
        "\tat [main] com.example.app.Widget.render(Widget.java:42)\n"
        "\tat task-connect-sink-0 do()\n"
    )
    ent = extract_entities(log)
    assert ent["modules"] == []  # ни один bracket-токен не просочился в модули


def test_extract_stoplist_filters_common():
    """org.apache.kafka.common — слишком широкий пакет, отсеивается; streams остаётся."""
    log = (
        "org.apache.kafka.common.errors.TimeoutException: Timeout\n"
        "\tat org.apache.kafka.streams.processor.internals.StreamThread.runOnce(StreamThread.java:900)\n"
    )
    ent = extract_entities(log)
    assert "common" not in ent["modules"]
    assert "streams" in ent["modules"]


# --- интеграция: сквозной обход ---

def _seed_impact_graph(conn):
    apply_schema(conn)
    load_records(
        conn,
        [
            node("Module", "module:clients", {"name": "clients"}),
            node("Module", "module:connect", {"name": "connect"}),
            # connect зависит от clients: падение clients затрагивает connect
            edge("DEPENDS_ON", "module:connect", "module:clients"),
            node("Task", "task:KAFKA-101",
                 {"key": "KAFKA-101", "summary": "NetworkClient reconnect loop",
                  "status": "Resolved", "uri": "https://issues/KAFKA-101"}),
            edge("MENTIONS", "task:KAFKA-101", "module:clients"),
            node("Person", "person:ann", {"name": "Ann Dev"}),
            edge("ASSIGNED_TO", "task:KAFKA-101", "person:ann"),
            node("Page", "page:1", {"title": "KIP-Networking", "uri": "https://wiki/net"}),
            edge("MENTIONS", "page:1", "task:KAFKA-101"),
        ],
    )


@pytest.mark.integration
def test_analyze_builds_impact_subgraph(neo4j_conn):
    _seed_impact_graph(neo4j_conn)
    analyzer = ImpactAnalyzer(neo4j_conn, max_hops=3)
    result = analyzer.analyze(SAMPLE_LOG)

    assert "module:clients" in result["failing"]
    affected_names = {m["name"] for m in result["affected_modules"]}
    assert "connect" in affected_names  # зависящий модуль затронут

    owners = {o["name"] for o in result["owners"]}
    assert "Ann Dev" in owners

    task_keys = {t["key"] for t in result["related_tasks"]}
    assert "KAFKA-101" in task_keys  # «уже чинили»

    page_uris = {p["uri"] for p in result["related_pages"]}
    assert "https://wiki/net" in page_uris


@pytest.mark.integration
def test_unmatched_entity_gives_empty_affected(neo4j_conn):
    """Сущность лога не матчится ни в один модуль -> честно пусто, не падение."""
    apply_schema(neo4j_conn)
    load_records(neo4j_conn, [node("Module", "module:streams", {"name": "streams"})])
    analyzer = ImpactAnalyzer(neo4j_conn)
    result = analyzer.analyze(SAMPLE_LOG)  # лог про clients, а есть только streams
    assert result["failing"] == []
    assert result["affected_modules"] == []


@pytest.mark.integration
def test_stub_task_without_key_excluded(neo4j_conn):
    """Заглушка-Task (без key, из git-ссылки) не попадает в related_tasks."""
    apply_schema(neo4j_conn)
    load_records(
        neo4j_conn,
        [
            node("Module", "module:clients", {"name": "clients"}),
            # реальный тикет
            node("Task", "task:KAFKA-101",
                 {"key": "KAFKA-101", "summary": "real", "status": "Resolved",
                  "uri": "https://issues/KAFKA-101"}),
            edge("MENTIONS", "task:KAFKA-101", "module:clients"),
            # заглушка без key/summary/uri, тоже упоминает модуль
            node("Task", "task:KAFKA-999", {}),
            edge("MENTIONS", "task:KAFKA-999", "module:clients"),
        ],
    )
    result = ImpactAnalyzer(neo4j_conn).impact_subgraph(["module:clients"])
    keys = {t["key"] for t in result["related_tasks"]}
    assert "KAFKA-101" in keys
    assert None not in keys  # заглушка отфильтрована
    assert all(t["key"] for t in result["related_tasks"])


@pytest.mark.integration
def test_traversal_respects_hop_limit(neo4j_conn):
    """Цепочка глубиной 4; при max_hops=3 самый дальний модуль не включается."""
    apply_schema(neo4j_conn)
    load_records(
        neo4j_conn,
        [
            node("Module", "module:clients", {"name": "clients"}),
            node("Module", "module:a", {"name": "a"}),
            node("Module", "module:b", {"name": "b"}),
            node("Module", "module:c", {"name": "c"}),
            node("Module", "module:d", {"name": "d"}),
            edge("DEPENDS_ON", "module:a", "module:clients"),  # хоп 1
            edge("DEPENDS_ON", "module:b", "module:a"),        # хоп 2
            edge("DEPENDS_ON", "module:c", "module:b"),        # хоп 3
            edge("DEPENDS_ON", "module:d", "module:c"),        # хоп 4
        ],
    )
    analyzer = ImpactAnalyzer(neo4j_conn, max_hops=3)
    result = analyzer.impact_subgraph(["module:clients"])
    names = {m["name"] for m in result["affected_modules"]}
    assert {"a", "b", "c"} <= names
    assert "d" not in names  # за пределом 3 хопов
