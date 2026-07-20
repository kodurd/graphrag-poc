"""Golden set из связей JIRA — почти бесплатная разметка.

Связи Duplicate / caused-by / MENTIONS дают эталонные пары «вопрос → нужные
узлы» без ручной разметки: на них считаются retrieval-метрики.
"""

from __future__ import annotations

from graphrag.graph.connection import Neo4jConnection
from graphrag.retrieval.hybrid import HybridRetriever

from eval.metrics import candidate_entity_id, precision_recall_f1


def build_from_graph(conn: Neo4jConnection, limit: int = 100) -> list[dict]:
    """Генерирует вопросы с эталонными множествами узлов из структуры графа."""
    items: list[dict] = []

    # «Похожая ошибка»: по summary тикета найти его дубликат.
    for r in conn.run(
        """
        MATCH (a:Task)-[:DUPLICATES]->(b:Task)
        WHERE a.summary IS NOT NULL AND a.summary <> ''
        RETURN a.summary AS q, b.id AS target LIMIT $lim
        """,
        lim=limit,
    ):
        items.append({"question": r["q"], "expected_ids": [r["target"]], "kind": "duplicate"})

    # «Тикеты про модуль X»: множество тикетов, упоминающих модуль.
    for r in conn.run(
        """
        MATCH (t:Task)-[:MENTIONS]->(m:Module)
        WITH m, collect(DISTINCT t.id) AS tasks
        WHERE size(tasks) > 0
        RETURN m.name AS mod, tasks LIMIT $lim
        """,
        lim=limit,
    ):
        items.append(
            {
                "question": f"какие тикеты касаются модуля {r['mod']}",
                "expected_ids": r["tasks"],
                "kind": "module",
            }
        )

    return items[:limit]


def evaluate_retrieval(retriever: HybridRetriever, golden: list[dict]) -> dict:
    """Прогоняет golden set через retriever, считает retrieval P/R/F1."""
    per_item = []
    for item in golden:
        result = retriever.retrieve(item["question"])
        got = {candidate_entity_id(c["id"]) for c in result["candidates"]}
        per_item.append(precision_recall_f1(got, item["expected_ids"]))
    n = len(per_item) or 1
    return {
        "n": len(per_item),
        "precision": sum(d["precision"] for d in per_item) / n,
        "recall": sum(d["recall"] for d in per_item) / n,
        "f1": sum(d["f1"] for d in per_item) / n,
    }
