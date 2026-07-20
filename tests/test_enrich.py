"""LLM-обогащение графа — парсер (чистый) + интеграция."""

from __future__ import annotations

import pytest

from graphrag.graph.enrich import Enricher, parse_enrichment
from graphrag.llm.base import LLMClient


def test_parse_enrichment_modules_and_depends():
    data = {
        "modules": [{"name": "auth-service"}, {"name": "gateway"}],
        "depends_on": [{"from": "gateway", "to": "auth-service"}],
    }
    recs = parse_enrichment(data)
    mods = {r["id"] for r in recs if r["kind"] == "node"}
    assert "module:auth-service" in mods and "module:gateway" in mods
    deps = [r for r in recs if r["kind"] == "edge" and r["type"] == "DEPENDS_ON"]
    assert deps and deps[0]["from"] == "module:gateway" and deps[0]["to"] == "module:auth-service"


def test_parse_enrichment_ignores_malformed_and_out_of_ontology():
    data = {
        "modules": [{"name": ""}, "goodname"],          # пустое отбрасывается, строка ок
        "depends_on": [{"from": "a"}, {"weird": "x"}],   # неполные связи отбрасываются
        "random_relation": [{"from": "a", "to": "b"}],   # вне онтологии — игнор
    }
    recs = parse_enrichment(data)
    assert any(r["id"] == "module:goodname" for r in recs)
    assert all(r.get("type") != "random_relation" for r in recs)
    assert not any(r["kind"] == "edge" for r in recs)  # ни одной валидной связи


def test_parse_enrichment_creates_stub_endpoints():
    """Концы depends_on появляются как Module, даже если не в списке modules."""
    recs = parse_enrichment({"depends_on": [{"from": "x", "to": "y"}]})
    mods = {r["id"] for r in recs if r["kind"] == "node"}
    assert mods == {"module:x", "module:y"}


@pytest.mark.integration
def test_enricher_writes_to_graph(neo4j_conn):
    from graphrag.graph.schema import apply_schema

    apply_schema(neo4j_conn)

    class ScriptedLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"modules": [{"name": "auth"}, {"name": "gateway"}], ' \
                   '"depends_on": [{"from": "gateway", "to": "auth"}]}'

    Enricher(neo4j_conn, ScriptedLLM("x")).enrich_text("любой текст про сервисы", "u1")

    rows = neo4j_conn.run(
        "MATCH (:Module {id:'module:gateway'})-[:DEPENDS_ON]->(:Module {id:'module:auth'}) "
        "RETURN count(*) AS c"
    )
    assert rows[0]["c"] == 1
