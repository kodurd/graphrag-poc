"""LLM-обогащение графа.

Извлекает сущности-модули и зависимости из текстов вики/тикетов по
ФИКСИРОВАННОЙ онтологии и мержит в скелет. Ограничение схемы —
ключ к качеству: LLM не даём выдумывать произвольные типы связей.
"""

from __future__ import annotations

from graphrag.graph.connection import Neo4jConnection
from graphrag.graph.skeleton import load_records
from graphrag.index.vector import collect_text_nodes
from graphrag.intermediate import edge, node
from graphrag.llm.base import LLMClient

ENRICH_PROMPT = (
    "Из текста извлеки сервисы/модули системы и зависимости между ними. "
    'Верни JSON: {"modules": [{"name": "..."}], '
    '"depends_on": [{"from": "модуль", "to": "модуль"}]}. '
    "Только JSON, только реально упомянутые сущности.\n\nТЕКСТ:\n"
)


def parse_enrichment(data: dict, *, source: dict | None = None) -> list[dict]:
    """Преобразует ответ LLM в node/edge-записи, отбрасывая всё вне онтологии."""
    if not isinstance(data, dict):
        return []
    src = source or {"source": "llm-enrich"}
    records: list[dict] = []
    seen: set[str] = set()

    def module_id(name: str) -> str:
        return f"module:{name.strip().lower()}"

    for m in data.get("modules", []) or []:
        name = (m.get("name") if isinstance(m, dict) else str(m)) or ""
        name = name.strip()
        if not name:
            continue
        mid = module_id(name)
        if mid not in seen:
            seen.add(mid)
            records.append(node("Module", mid, {"name": name}, src))

    for d in data.get("depends_on", []) or []:
        if not isinstance(d, dict):
            continue
        a, b = (d.get("from") or "").strip(), (d.get("to") or "").strip()
        if not a or not b:
            continue
        # гарантируем существование концов как Module (заглушки при необходимости)
        for name in (a, b):
            mid = module_id(name)
            if mid not in seen:
                seen.add(mid)
                records.append(node("Module", mid, {"name": name}, src))
        records.append(edge("DEPENDS_ON", module_id(a), module_id(b), source=src))

    return records


class Enricher:
    """Прогоняет тексты графа через LLM и мержит извлечённое обратно."""

    def __init__(self, conn: Neo4jConnection, llm: LLMClient):
        self.conn = conn
        self.llm = llm

    def enrich_text(self, text: str, source_uri: str | None = None) -> dict:
        data = self.llm.extract_json(ENRICH_PROMPT + text[:4000])
        records = parse_enrichment(
            data if isinstance(data, dict) else {},
            source={"source": "llm-enrich", "uri": source_uri or ""},
        )
        if records:
            load_records(self.conn, records)
        return {"records": len(records)}

    def enrich_graph(self, limit: int | None = None) -> dict:
        """Обогащает по текстам всех текстонесущих узлов (Task/Commit/Page)."""
        nodes = collect_text_nodes(self.conn)
        if limit:
            nodes = nodes[:limit]
        total = 0
        failed = 0
        for _id, text, uri in nodes:
            # Один сбойный узел (сеть/невалидный JSON) не должен ронять весь прогон —
            # уже записанное идемпотентно, но перезапуск заново оплачивал бы всё.
            try:
                total += self.enrich_text(text, uri)["records"]
            except Exception:
                failed += 1
        return {"enriched_nodes": len(nodes), "records": total, "failed": failed}
