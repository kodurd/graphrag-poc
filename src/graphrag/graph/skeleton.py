"""Загрузка детерминированного скелета в Neo4j из промежуточного JSONL.

Идемпотентно (MERGE): повторный прогон не плодит дубли. Концы рёбер, для
которых нет node-записи (напр. MENTIONS на тикет вне среза), создаются как
узлы-заглушки — обход графа на них не обрывается.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from graphrag.graph.connection import Neo4jConnection
from graphrag.graph.schema import label_of, validate_label, validate_rel
from graphrag.intermediate import read_jsonl


def plan_writes(
    records: Iterable[dict],
) -> tuple[dict[str, list[dict]], dict[tuple[str, str, str], list[dict]]]:
    """Группирует записи в план записи (чистая функция, без Neo4j).

    Возвращает (nodes_by_label, edges_by_key), где key = (from_label, rel, to_label).
    """
    nodes_by_label: dict[str, list[dict]] = defaultdict(list)
    edges_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for r in records:
        if r["kind"] == "node":
            props = dict(r.get("props") or {})
            props["id"] = r["id"]
            src = r.get("source") or {}
            if src.get("uri") and "uri" not in props:
                props["uri"] = src["uri"]
            nodes_by_label[validate_label(r["label"])].append(props)
        elif r["kind"] == "edge":
            fl = label_of(r["from"])
            tl = label_of(r["to"])
            rel = validate_rel(r["type"])
            edges_by_key[(fl, rel, tl)].append(
                {"from": r["from"], "to": r["to"], "props": r.get("props") or {}}
            )

    return dict(nodes_by_label), dict(edges_by_key)


def load_records(conn: Neo4jConnection, records: Iterable[dict]) -> dict:
    """MERGE-ит записи в граф. Возвращает статистику."""
    nodes_by_label, edges_by_key = plan_writes(records)
    stats = {"nodes": 0, "edges": 0}

    for label, rows in nodes_by_label.items():
        conn.run(
            f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
            rows=rows,
        )
        stats["nodes"] += len(rows)

    for (fl, rel, tl), rows in edges_by_key.items():
        conn.run(
            f"""
            UNWIND $rows AS row
            MERGE (a:{fl} {{id: row.from}})
            MERGE (b:{tl} {{id: row.to}})
            MERGE (a)-[e:{rel}]->(b)
            SET e += row.props
            """,
            rows=rows,
        )
        stats["edges"] += len(rows)

    return stats


def load_jsonl(conn: Neo4jConnection, *paths: str) -> dict:
    """Загружает один или несколько JSONL-файлов промежуточного хранилища."""
    def _all():
        for p in paths:
            yield from read_jsonl(p)

    return load_records(conn, _all())
