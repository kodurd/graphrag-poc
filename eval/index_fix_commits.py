"""Индексация фикс-связанных коммитов: делает how-to-фиксы достижимыми ретривом.

Замер `fix_linkage` показал: коммиты-фиксы в графе и связаны с тикетами
(`Commit−MENTIONS→Task`), но НЕ в поисковом индексе (0 коммит-чанков) → недостижимы.
Этот раннер собирает только фикс-связанные коммиты и грузит их существующим
`VectorIndexer` (идемпотентно, resume пропускает готовые чанки).

Запуск (нужны --extra ml + Neo4j):
    uv run --extra ml python -m eval.index_fix_commits
"""

from __future__ import annotations


def main() -> int:
    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder
    from graphrag.graph import Neo4jConnection
    from graphrag.index.vector import VectorIndexer, collect_fix_commit_nodes

    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("index-fix-commits: Neo4j недоступен — `docker compose up -d`")
            return 1

        before = conn.run(
            "MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk:commit:' RETURN count(c) AS n"
        )[0]["n"]
        print(f"index-fix-commits: коммит-чанков до прогона: {before}", flush=True)

        nodes = collect_fix_commit_nodes(conn)
        print(f"index-fix-commits: фикс-связанных коммитов {len(nodes)}", flush=True)

        embedder = build_embedder(s.embeddings)
        indexer = VectorIndexer(
            conn, embedder, size=s.chunk.size, overlap=s.chunk.overlap
        )
        indexer.ensure_index()
        stats = indexer.index_nodes(nodes, batch_size=64, progress=True)
        conn.run("CALL db.awaitIndexes(300)")

        after = conn.run(
            "MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk:commit:' RETURN count(c) AS n"
        )[0]["n"]
        print(
            f"DONE index-fix-commits: {stats} commit-chunks {before}->{after} "
            f"(embedder={s.embeddings.provider})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
