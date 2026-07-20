"""Векторный индекс поверх Neo4j: чанки как узлы с эмбеддингами.

Каждый Chunk хранит text, uri (для цитирования) и embedding; привязан к
родителю ребром PART_OF. Поиск — через встроенный vector index Neo4j.
"""

from __future__ import annotations

from graphrag.embeddings.base import Embedder
from graphrag.graph.connection import Neo4jConnection
from graphrag.index.chunk import plan_chunks

# Какие узлы и как дают текст для чанкинга (label, cypher-выражение текста).
TEXT_SOURCES: list[tuple[str, str]] = [
    ("Task", "coalesce(n.summary,'') + '\\n' + coalesce(n.description,'')"),
    ("Commit", "coalesce(n.message,'')"),
    ("Page", "coalesce(n.title,'') + '\\n' + coalesce(n.text,'')"),
]


def collect_text_nodes(
    conn: Neo4jConnection, labels: list[str] | None = None
) -> list[tuple[str, str, str | None]]:
    """Собирает (id, text, uri) из текстонесущих узлов графа.

    labels ограничивает типы (напр. ['Task', 'Page']) — на большом корпусе
    Commit'ов на порядок больше остальных, и их индексация может быть не нужна.
    """
    out: list[tuple[str, str, str | None]] = []
    for label, text_expr in TEXT_SOURCES:
        if labels and label not in labels:
            continue
        rows = conn.run(
            f"MATCH (n:{label}) RETURN n.id AS id, {text_expr} AS text, n.uri AS uri"
        )
        for r in rows:
            if (r.get("text") or "").strip():
                out.append((r["id"], r["text"], r.get("uri")))
    return out


class VectorIndexer:
    """Строит и опрашивает векторный индекс чанков."""

    def __init__(
        self,
        conn: Neo4jConnection,
        embedder: Embedder,
        *,
        size: int = 800,
        overlap: int = 120,
        index_name: str = "chunk_embedding",
    ):
        self.conn = conn
        self.embedder = embedder
        self.size = size
        self.overlap = overlap
        self.index_name = index_name

    def ensure_index(self) -> None:
        self.conn.ensure_vector_index(
            self.index_name, "Chunk", "embedding", self.embedder.dimension
        )

    _MERGE_CHUNKS = """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.id})
        SET c.text = row.text, c.uri = row.uri, c.seq = row.seq, c.parent = row.parent
        WITH c, row
        CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
        WITH c, row
        MATCH (p {id: row.parent})
        MERGE (c)-[:PART_OF]->(p)
        """

    def index_nodes(
        self,
        nodes: list[tuple[str, str, str | None]],
        *,
        batch_size: int = 64,
        progress: bool = False,
    ) -> dict:
        """Чанкует тексты, считает эмбеддинги и грузит Chunk-узлы.

        Идёт пачками и сохраняет после каждой: память ограничена размером батча,
        а прерванный прогон продолжается с места остановки — уже посчитанные
        чанки пропускаются (resume).
        """
        specs = plan_chunks(nodes, self.size, self.overlap)
        if not specs:
            return {"chunks": 0, "skipped": 0}

        done_ids = {
            r["id"]
            for r in self.conn.run("MATCH (c:Chunk) RETURN c.id AS id")
        }
        todo = [s for s in specs if s["id"] not in done_ids]
        skipped = len(specs) - len(todo)

        written = 0
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            vectors = self.embedder.encode([s["text"] for s in batch])
            rows = [
                {
                    "id": s["id"],
                    "parent": s["parent"],
                    "text": s["text"],
                    "uri": s["uri"],
                    "seq": s["seq"],
                    "embedding": vectors[i].tolist(),
                }
                for i, s in enumerate(batch)
            ]
            self.conn.run(self._MERGE_CHUNKS, rows=rows)
            written += len(rows)
            if progress:
                print(f"  chunks {written}/{len(todo)} (skipped {skipped})", flush=True)

        return {"chunks": written, "skipped": skipped}

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Возвращает ближайшие чанки: [{id, text, uri, score}]."""
        qvec = self.embedder.encode([query])[0].tolist()
        return self.conn.run(
            """
            CALL db.index.vector.queryNodes($name, $k, $vec) YIELD node, score
            RETURN node.id AS id, node.text AS text, node.uri AS uri, score
            ORDER BY score DESC
            """,
            name=self.index_name,
            k=top_k,
            vec=qvec,
        )
