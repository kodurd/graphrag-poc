"""Онтология графа и применение схемы в Neo4j.

Фиксированный набор меток и типов рёбер — качество извлечения и предсказуемость
обходов резко растут против свободной онтологии.
"""

from __future__ import annotations

from graphrag.graph.connection import Neo4jConnection

NODE_LABELS: tuple[str, ...] = (
    "Commit",
    "File",
    "Module",
    "Task",
    "Person",
    "Page",
    "Error",
    "Chunk",
)

EDGE_TYPES: tuple[str, ...] = (
    "DEPENDS_ON",
    "IMPORTS",
    "MENTIONS",
    "ASSIGNED_TO",
    "LINKS_TO",
    "TOUCHES",
    "OCCURRED_IN",
    "FIXED_BY",
    "DUPLICATES",
    "PART_OF",
)

# id-префикс → метка узла (для разрешения концов рёбер и заглушек).
PREFIX_TO_LABEL: dict[str, str] = {
    "commit": "Commit",
    "file": "File",
    "module": "Module",
    "task": "Task",
    "person": "Person",
    "page": "Page",
    "error": "Error",
    "chunk": "Chunk",
}


def label_of(node_id: str) -> str:
    """Метка узла по префиксу его id ('task:KAFKA-1' -> 'Task')."""
    prefix = node_id.split(":", 1)[0]
    return PREFIX_TO_LABEL.get(prefix, "Entity")


def validate_label(label: str) -> str:
    if label not in NODE_LABELS and label != "Entity":
        raise ValueError(f"неизвестная метка узла: {label!r}")
    return label


def validate_rel(rel: str) -> str:
    if rel not in EDGE_TYPES:
        raise ValueError(f"неизвестный тип ребра: {rel!r}")
    return rel


def apply_schema(conn: Neo4jConnection) -> None:
    """Идемпотентно создаёт constraints уникальности id для всех меток."""
    for label in NODE_LABELS:
        conn.run(
            f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
