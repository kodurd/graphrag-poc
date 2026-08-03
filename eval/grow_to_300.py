"""Разовый рост набора оценки до ~300 (доверие при малом n).

Берёт ТЕКУЩИЙ questions_grown (n=104) как базу, генерит новые из графа, сливает
с дедупом, обрезает до TARGET. Пишет обратно questions_grown.json. Baseline
n=104 уже сохранён в questions_grown_n104.json.
"""
from __future__ import annotations

import json

from graphrag.config import load_settings
from graphrag.graph import Neo4jConnection
from graphrag.llm import build_llm

from eval.grow_set import merge_dedup
from eval.question_gen import generate_from_graph

T = "eval/trial"
TARGET = 300
GEN_LIMIT = 300  # генерим с запасом; часть отсеется дедупом против существующих


def main() -> int:
    existing = json.load(open(f"{T}/questions_grown.json", encoding="utf-8"))
    print(f"grow-300: старт с {len(existing)} вопросов", flush=True)

    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        llm = build_llm(s.llm, role="generation")
        generated = generate_from_graph(conn, llm, limit=GEN_LIMIT)
    print(f"grow-300: сгенерировано {len(generated)}", flush=True)

    merged = merge_dedup(existing, generated)[:TARGET]
    json.dump(merged, open(f"{T}/questions_grown.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"DONE grow-300: {len(existing)} + {len(generated)} -> {len(merged)} "
          f"(cap {TARGET}) -> {T}/questions_grown.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
