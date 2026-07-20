"""Гибридный retrieval с маршрутизацией (Слой 3).

Объединяет три источника кандидатов — векторный поиск, BM25 и обход графа —
по маршруту интента, дедуплицирует и переранжирует cross-encoder'ом.
Ключевая демонстрация: multi-hop-вопрос через граф находит связи,
которых вектор-only не видит.
"""

from __future__ import annotations

from graphrag.embeddings.base import Embedder, Reranker
from graphrag.graph.connection import Neo4jConnection
from graphrag.index.bm25 import BM25Index
from graphrag.index.vector import VectorIndexer
from graphrag.llm.base import LLMClient
from graphrag.retrieval.router import FACTUAL, MIXED, MULTIHOP, classify_intent
from graphrag.text import tokenize


class GraphRetriever:
    """Кандидаты из обхода графа: модули, связанные зависимостями с запросом."""

    def __init__(self, conn: Neo4jConnection, max_hops: int = 2):
        self.conn = conn
        self.max_hops = max(1, min(int(max_hops), 5))

    def _module_names_in_query(self, query: str) -> list[str]:
        tokens = set(tokenize(query))
        rows = self.conn.run("MATCH (m:Module) RETURN toLower(m.name) AS name")
        return [r["name"] for r in rows if r["name"] and r["name"] in tokens]

    def related(self, query: str) -> list[dict]:
        names = self._module_names_in_query(query)
        if not names:
            return []
        hops = self.max_hops
        rows = self.conn.run(
            f"""
            MATCH (m:Module) WHERE toLower(m.name) IN $names
            MATCH (rel:Module)-[:DEPENDS_ON*1..{hops}]-(m)
            WHERE NOT toLower(rel.name) IN $names
            RETURN DISTINCT rel.id AS id, rel.name AS name
            ORDER BY name
            """,
            names=names,
        )
        joined = ", ".join(names)
        return [
            {
                "id": r["id"],
                "text": f"Модуль {r['name']} связан зависимостями с: {joined}",
                # citable-ссылка на узел графа: иначе build_context (требующий uri)
                # выбросит graph-факты, и чисто multi-hop ask вернёт «нет данных».
                "uri": f"graph://{r['id']}",
            }
            for r in rows
        ]


def filter_by_threshold(items: list[dict], min_score: float) -> list[dict]:
    """Отбрасывает вектор/bm25-кандидатов ниже порога reranker-скора.

    Граф-кандидаты (`source == "graph"`) освобождены: их текст синтетический и
    лексически скорится низко — порог выкосил бы graph-only модули.
    Порог 0 (или меньше) — фильтр отключён, поведение как раньше.
    """
    if min_score <= 0:
        return items
    return [
        it for it in items
        if it.get("source") == "graph" or it.get("rerank_score", 0.0) >= min_score
    ]


class HybridRetriever:
    """Маршрутизирует запрос по источникам, объединяет и переранжирует."""

    def __init__(
        self,
        conn: Neo4jConnection,
        embedder: Embedder,
        reranker: Reranker,
        *,
        llm: LLMClient | None = None,
        top_k: int = 8,
        rerank_top_k: int = 5,
        max_hops: int = 2,
        min_rerank_score: float = 0.0,
    ):
        self.conn = conn
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.min_rerank_score = min_rerank_score
        self.vector = VectorIndexer(conn, embedder)
        self.graph = GraphRetriever(conn, max_hops)
        self._bm25: BM25Index | None = None

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            docs = self.conn.run(
                "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text, c.uri AS uri"
            )
            self._bm25 = BM25Index(docs)
        return self._bm25

    def retrieve(self, query: str) -> dict:
        # Модули, упомянутые в запросе — из того же графового источника, что и обход
        # (переиспользуем `_module_names_in_query`, покрывающий все узлы Module, а не
        # только corpus.components). Пусто => impact-вопрос без модуля => MIXED.
        known = self.graph._module_names_in_query(query)
        route = classify_intent(query, self.llm, known_modules=known)
        merged: dict[str, dict] = {}

        def add(items: list[dict], source: str) -> None:
            for it in items:
                merged.setdefault(it["id"], {**it, "source": source})

        if route in (FACTUAL, MIXED):
            add(self.vector.search(query, self.top_k), "vector")
            add(self._bm25_index().search(query, self.top_k), "bm25")
        if route in (MULTIHOP, MIXED):
            add(self.graph.related(query), "graph")

        items = list(merged.values())
        if items:
            ranked = self.reranker.rerank(query, [it["text"] for it in items])
            items = [{**items[i], "rerank_score": score} for i, score in ranked]
            items = filter_by_threshold(items, self.min_rerank_score)

        return {"route": route, "candidates": items[: self.rerank_top_k]}
