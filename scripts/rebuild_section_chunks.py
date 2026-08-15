"""Пере-нарезка Page-чанков по разделам (## ) — честный AFTER для section-A/B.

Опасный шаг (delete + re-embed), поэтому защищённый:
1. Размечает Page.text из confluence.jsonl (load_jsonl, идемпотентно). Проверяет,
   что KIP-страниц с '## ' стало >= MARK_FLOOR — ИНАЧЕ ABORT до всякого удаления
   (не рушим рабочий блёклый граф без замены). 81 страница из старого бэкапа не
   размечается (нет HTML) — останутся блёклыми, это ожидаемо (~86% покрытие).
2. Удаляет chunk:page:* (иначе дедуп по id не даст пересоздать секционные чанки).
3. Пере-индексирует Page-узлы: размеченные -> секционно, остальные -> блёкло.
4. Fingerprint + проверка покрытия.

Блёклый бэкап (neo4j_with-kip-complete-blind) — страховка отката при обрыве.
Запуск: uv run --extra ml python scripts/rebuild_section_chunks.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag.config import load_settings
from graphrag.embeddings import build_embedder
from graphrag.graph import Neo4jConnection
from graphrag.graph.skeleton import load_jsonl
from graphrag.index.vector import VectorIndexer, collect_text_nodes

from eval.measure_guard import graph_fingerprint

MARK_FLOOR = 400   # ожидаем ~490 размеченных KIP; ниже — разметка не сработала, ABORT


def _kip_marked(conn) -> int:
    return list(conn.run(
        "MATCH (p:Page) WHERE p.title STARTS WITH 'KIP-' AND p.text CONTAINS '## ' "
        "RETURN count(p) AS x"))[0]["x"]


def _kip_with_chunks(conn) -> int:
    return list(conn.run(
        "MATCH (p:Page)<-[:PART_OF]-(:Chunk) WHERE p.title STARTS WITH 'KIP-' "
        "RETURN count(DISTINCT p) AS x"))[0]["x"]


def main() -> int:
    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("rebuild-section: Neo4j недоступен"); return 1

        print(f"BEFORE fingerprint: {graph_fingerprint(conn)}", flush=True)

        # 1) разметка Page.text
        paths = sorted(glob.glob("data/intermediate/confluence.jsonl"))
        stats = load_jsonl(conn, *paths)
        marked = _kip_marked(conn)
        print(f"load_jsonl {stats} -> KIP размечено: {marked}", flush=True)
        if marked < MARK_FLOOR:
            print(f"ABORT: размечено {marked} < {MARK_FLOOR} — разметка не сработала, "
                  "чанки НЕ удаляю (граф цел).", flush=True)
            return 3

        # 2) снести старые page-чанки (иначе дедуп заблокирует пересоздание)
        conn.run("MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk:page:' DETACH DELETE c")
        left = list(conn.run(
            "MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk:page:' RETURN count(c) AS x"))[0]["x"]
        print(f"удалено chunk:page — осталось {left}", flush=True)

        # 3) пере-индексация Page (размеченные режутся секционно в chunk.py)
        indexer = VectorIndexer(conn, build_embedder(s.embeddings),
                                size=s.chunk.size, overlap=s.chunk.overlap)
        indexer.ensure_index()
        istats = indexer.index_nodes(collect_text_nodes(conn, ["Page"]),
                                     batch_size=64, progress=True)
        print(f"index_nodes: {istats}", flush=True)

        fp = graph_fingerprint(conn)
        kc = _kip_with_chunks(conn)
        print(f"AFTER fingerprint: {fp}", flush=True)
        print(f"AFTER KIP-страниц с чанками: {kc}/571 · размечено секц: {fp['kip_pages_sectioned']}",
              flush=True)
        ok = kc >= 560 and fp["kip_pages_sectioned"] >= MARK_FLOOR
        print(f"VERDICT: {'OK' if ok else 'CHECK'} (секц.страниц {fp['kip_pages_sectioned']}, "
              f"с чанками {kc})", flush=True)
        print("DONE rebuild-section", flush=True)
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
