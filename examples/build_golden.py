"""Собрать наш golden set из текущего графа и сохранить снапшот.

`build_from_graph` выводит эталонные пары «вопрос -> нужные узлы» из структуры
графа (DUPLICATES между тикетами, MENTIONS тикет->модуль) — почти бесплатная
разметка. Здесь мы фиксируем результат в версионируемый JSON, чтобы eval был
воспроизводим независимо от состояния БД.

Запуск (после seed_demo.py): uv run python examples/build_golden.py
Затем метрики: uv run graphrag eval   (нужен построенный вектор-индекс)
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag.config import load_settings
from graphrag.graph import Neo4jConnection

from eval.golden_set import build_from_graph

OUT_PATH = Path("eval/golden_demo.json")


def main() -> None:
    settings = load_settings()
    with Neo4jConnection(settings.neo4j) as conn:
        golden = build_from_graph(conn, limit=200)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_kind: dict[str, int] = {}
    for g in golden:
        by_kind[g["kind"]] = by_kind.get(g["kind"], 0) + 1
    print(f"golden: {len(golden)} items -> {OUT_PATH}  ({by_kind})")
    for g in golden:
        print(f"  [{g['kind']:9}] {g['question']!r} -> {len(g['expected_ids'])} expected")


if __name__ == "__main__":
    main()
