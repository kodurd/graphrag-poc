"""Бэкфилл графа обогащёнными коммитами (тело %B) + новые MENTIONS по телу.

Commits-only: обновляет Commit.message на месте и добавляет body-производные
MENTIONS (`ref_source='body'`), НЕ пересканируя Java/File/Module (дёшево).
Перед прогоном снимает снапшот старых message для отката (MERGE перезаписывает
message необратимо).

Запуск (нужен Neo4j + локальный клон Kafka):
    uv run python -m eval.backfill_commit_bodies
"""

from __future__ import annotations

import json
from pathlib import Path

_SNAPSHOT = "eval/trial/commit_message_snapshot.jsonl"
_INTERMEDIATE = "data/intermediate/git_commits_body.jsonl"


def _mentions_counts(conn) -> tuple[int, int]:
    total = conn.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n")[0]["n"]
    body = conn.run(
        "MATCH ()-[m:MENTIONS]->() WHERE m.ref_source = 'body' RETURN count(m) AS n"
    )[0]["n"]
    return total, body


def main() -> int:
    from graphrag.config import load_settings
    from graphrag.connectors.git import GitConnector
    from graphrag.graph import Neo4jConnection
    from graphrag.graph.skeleton import load_jsonl

    s = load_settings()
    if not Path(s.corpus.repo_path).exists():
        print(f"backfill: репозиторий не найден: {s.corpus.repo_path}")
        return 1

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("backfill: Neo4j недоступен — `docker compose up -d`")
            return 1

        # Снапшот старых message (откат): MERGE ниже перезапишет message необратимо.
        rows = conn.run("MATCH (c:Commit) RETURN c.id AS id, c.message AS message")
        Path(_SNAPSHOT).parent.mkdir(parents=True, exist_ok=True)
        with Path(_SNAPSHOT).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"id": r["id"], "message": r["message"]},
                                    ensure_ascii=False) + "\n")
        print(f"backfill: снапшот {len(rows)} старых message -> {_SNAPSHOT}", flush=True)

        m_total_before, m_body_before = _mentions_counts(conn)
        print(f"backfill: MENTIONS до: всего {m_total_before}, body {m_body_before}", flush=True)

        # commits-only эмит (обогащённое тело) -> JSONL -> MERGE-загрузка.
        Path(_INTERMEDIATE).parent.mkdir(parents=True, exist_ok=True)
        gc = GitConnector(s.corpus.repo_path, s.corpus.components, s.corpus.since)
        estats = gc.extract_commits_only(_INTERMEDIATE)
        print(f"backfill: эмит {estats}", flush=True)
        lstats = load_jsonl(conn, _INTERMEDIATE)
        print(f"backfill: загрузка {lstats}", flush=True)

        m_total_after, m_body_after = _mentions_counts(conn)
        # Пример обогащённого message (самый длинный)
        sample = conn.run(
            "MATCH (c:Commit) RETURN c.id AS id, c.message AS message "
            "ORDER BY size(c.message) DESC LIMIT 1"
        )
        s0 = sample[0] if sample else {}
        print(
            f"DONE backfill: MENTIONS всего {m_total_before}->{m_total_after} "
            f"(body {m_body_before}->{m_body_after}); "
            f"пример message len={len(s0.get('message','') or '')}",
            flush=True,
        )
        print(f"  [{s0.get('id')}] {str(s0.get('message',''))[:200]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
