"""Снимок качества на фиксированном наборе вопросов (не авто-генерация).

`eval-quality` CLI генерирует вопросы из графа заново на каждом прогоне — числа
между прогонами несопоставимы. Этот раннер берёт **зафиксированный** набор
(`eval/trial/questions_grown.json`, n=104) и прогоняет его текущим прод-конвейером
(cross-encoder reranker + multihop full-retrieval), чтобы получить публикуемый,
воспроизводимый снимок reference-free метрик.

Запуск (нужен Neo4j с загруженным корпусом, LLM-ключ, extra ml):
    uv run --extra ml python -m eval.quality_snapshot

Артефакты (utf-8): eval/quality_snapshot_report.md + eval/quality_snapshot_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from graphrag.config import load_settings
from graphrag.embeddings import build_embedder, build_reranker
from graphrag.graph import Neo4jConnection
from graphrag.llm import build_llm
from graphrag.retrieval.hybrid import HybridRetriever

from eval.quality_eval import evaluate_question
from eval.quality_report import render_report

_QUESTIONS = Path("eval/trial/questions_grown.json")
_REPORT = Path("eval/quality_snapshot_report.md")
_RESULTS = Path("eval/quality_snapshot_results.json")


def main() -> int:
    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        print("quality-snapshot: не задан LLM_API_KEY (.env).")
        return 1

    questions = json.loads(_QUESTIONS.read_text(encoding="utf-8"))
    print(
        f"quality-snapshot: вопросов {len(questions)} · reranker "
        f"{s.reranker.provider} ({s.reranker.model}) · multihop_full="
        f"{s.retrieval.multihop_full_retrieval}",
        flush=True,
    )

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("quality-snapshot: Neo4j недоступен — `docker compose up -d`")
            return 1

        llm = build_llm(s.llm, role="generation")
        retr = HybridRetriever(
            conn,
            build_embedder(s.embeddings),
            build_reranker(s.reranker),
            top_k=s.retrieval.top_k,
            rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops,
            min_rerank_score=s.retrieval.min_rerank_score,
        )

        # Прогресс по ходу: длинный прогон, хочется видеть, что он жив.
        records: list[dict] = []
        for i, item in enumerate(questions, 1):
            try:
                records.append(
                    evaluate_question(
                        retr, llm, item["question"], source_id=item.get("source_id")
                    )
                )
            except Exception as e:  # сетевой сбой на одном вопросе не роняет прогон
                print(f"  [{i}/{len(questions)}] SKIP: {type(e).__name__}: {e}", flush=True)
                continue
            if i % 5 == 0 or i == len(questions):
                print(f"  [{i}/{len(questions)}] готово", flush=True)

    results = {
        "records": records,
        "counts": {"questions": len(records), "labeled": 0, "total": len(records)},
    }
    _RESULTS.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(results)
    _REPORT.write_text(report, encoding="utf-8")
    print(f"quality-snapshot: отчёт -> {_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
