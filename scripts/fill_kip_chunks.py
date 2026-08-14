"""Досоздать недостающие чанки Page-узлов (в т.ч. KIP) БЕЗ перезагрузки jsonl.

Регрессия: в графе у 473/571 KIP-страниц есть текст, но нет чанков (билд убивали
на середине эмбеддинга). Из-за этого вектор-поиск не достаёт KIP -> answer-rate
упал к до-KIP уровню (54% вместо 63.7%).

Чинит точечно: чанкует Page-узлы из ТЕКУЩЕГО текста графа (он не размечен -> чанки
блёклые, однородно с уже готовыми 98 страницами; воспроизводит здоровый baseline,
а НЕ вводит секционный чанкинг — это отдельный эксперимент). index_nodes пропускает
существующие чанки по id, поэтому embed идёт только для недостающих (resume-safe).

Запуск: uv run --extra ml python -m scripts.fill_kip_chunks
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo-root в path, чтобы импортировался пакет eval (scripts не является пакетом)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag.config import load_settings
from graphrag.embeddings import build_embedder
from graphrag.graph import Neo4jConnection
from graphrag.index.vector import VectorIndexer, collect_text_nodes

from eval.measure_guard import graph_fingerprint


def _kip_pages_with_chunks(conn) -> int:
    q = ("MATCH (p:Page)<-[:PART_OF]-(:Chunk) WHERE p.title STARTS WITH 'KIP-' "
         "RETURN count(DISTINCT p) AS x")
    return list(conn.run(q))[0]["x"]


def main() -> int:
    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("fill-kip: Neo4j недоступен")
            return 1

        fp_before = graph_fingerprint(conn)
        kip_before = _kip_pages_with_chunks(conn)
        print(f"BEFORE fingerprint: {fp_before}", flush=True)
        print(f"BEFORE KIP-страниц с чанками: {kip_before}/571", flush=True)

        # Только Page-узлы: недостающие чанки — у Confluence/KIP-страниц; Task/Commit
        # уже проиндексированы, их re-plan лишний и дорогой.
        nodes = collect_text_nodes(conn, ["Page"])
        print(f"Page-узлов с текстом собрано: {len(nodes)}", flush=True)

        indexer = VectorIndexer(
            conn, build_embedder(s.embeddings),
            size=s.chunk.size, overlap=s.chunk.overlap,
        )
        indexer.ensure_index()
        stats = indexer.index_nodes(nodes, batch_size=64, progress=True)
        print(f"index_nodes: {stats}", flush=True)

        fp_after = graph_fingerprint(conn)
        kip_after = _kip_pages_with_chunks(conn)
        print(f"AFTER fingerprint: {fp_after}", flush=True)
        print(f"AFTER KIP-страниц с чанками: {kip_after}/571", flush=True)

        ok = kip_after >= 560   # почти все 571 (допуск на пустые/битые)
        print(f"VERDICT: {'OK' if ok else 'INCOMPLETE'} "
              f"(KIP {kip_before}->{kip_after}, новых чанков {stats['chunks']})", flush=True)
        print("DONE fill-kip", flush=True)
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
