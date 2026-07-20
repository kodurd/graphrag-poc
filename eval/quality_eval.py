"""Прогонный харнесс оценки качества ответов.

По каждому вопросу: retrieve -> контекст -> генерация ответа -> судьи. Собирает
per-question записи с метриками и метаданными (маршрут, источники, grounded).

Устройство под бюджет и устойчивость:
- прогон **последовательный**: без параллелизма;
- сбой отдельного судьи даёт `None` и не роняет прогон — такие значения
  исключаются из знаменателя своей метрики при агрегации;
- retrieval P/R/F1 считается **отдельно** на графовом golden set: у авто-вопросов
  нет `expected_ids`, которых требует `evaluate_retrieval`, поэтому это другая
  популяция вопросов и отдельный блок результата.
"""

from __future__ import annotations

from graphrag.generate.answer import build_context, generate_answer
from graphrag.llm.base import LLMClient

from eval.metrics import (
    judge_answer_correctness,
    judge_answer_relevance,
    judge_context_precision,
    judge_context_recall,
    judge_faithfulness,
)

# Метрики, считающиеся без эталона, и метрики, которым эталон нужен.
REFERENCE_FREE = ("faithfulness", "answer_relevance", "context_precision")
REFERENCE_REQUIRED = ("answer_correctness", "context_recall")


def evaluate_question(
    retriever,
    llm: LLMClient,
    question: str,
    *,
    reference: str | None = None,
    source_id: str | None = None,
) -> dict:
    """Прогоняет один вопрос и возвращает запись с метриками и метаданными.

    `reference` задаётся только для размеченного среза — тогда дополнительно
    считаются correctness и context recall.
    """
    retrieved = retriever.retrieve(question)
    candidates = retrieved.get("candidates", [])
    context = build_context(candidates)
    context_texts = [c.get("text", "") for c in candidates]

    answer = generate_answer(llm, question, context)

    # faithfulness несёт флаг воздержания — распаковываем: число в metrics
    # (остаётся float|None), флаг — в sibling-поле ВНЕ metrics (инвариант metrics:
    # все значения число либо None). abstained=True => воздержание, False => сбой/оценка.
    faith_score, faith_abstained = judge_faithfulness(llm, answer.text, context_texts)

    record: dict = {
        "question": question,
        "source_id": source_id,
        "route": retrieved.get("route"),
        "answer": answer.text,
        "citations": answer.citations,
        "grounded": answer.grounded,
        "context_ids": [c.get("id") for c in candidates],
        "metrics": {
            "faithfulness": faith_score,
            "answer_relevance": judge_answer_relevance(llm, question, answer.text),
            "context_precision": judge_context_precision(llm, question, context_texts),
        },
        "abstained": {"faithfulness": faith_abstained},
    }

    if reference:
        record["reference"] = reference
        record["metrics"]["answer_correctness"] = judge_answer_correctness(
            llm, question, answer.text, reference
        )
        record["metrics"]["context_recall"] = judge_context_recall(
            llm, reference, context_texts
        )

    return record


def run_quality_eval(
    retriever,
    llm: LLMClient,
    questions: list[dict],
    labeled: list[dict] | None = None,
) -> dict:
    """Последовательный прогон набора вопросов и размеченного среза.

    `questions` — записи {question, source_id}; `labeled` — записи
    {question, reference, source_id}. Возвращает
    {"records": [...], "counts": {...}}.
    """
    records: list[dict] = []

    for item in questions:
        records.append(
            evaluate_question(
                retriever,
                llm,
                item["question"],
                source_id=item.get("source_id"),
            )
        )

    for item in labeled or []:
        records.append(
            evaluate_question(
                retriever,
                llm,
                item["question"],
                reference=item.get("reference"),
                source_id=item.get("source_id"),
            )
        )

    return {
        "records": records,
        "counts": {
            "questions": len(questions),
            "labeled": len(labeled or []),
            "total": len(records),
        },
    }


def run_retrieval_eval(retriever, golden: list[dict]) -> dict:
    """Retrieval P/R/F1 на графовом golden set — отдельная популяция вопросов.

    Вынесено из per-question записей намеренно: golden-вопросы порождены графом
    (`build_from_graph`) и несут `expected_ids`, которых у авто-вопросов нет.
    """
    from eval.golden_set import evaluate_retrieval

    report = evaluate_retrieval(retriever, golden)
    return {"population": "graph-golden", **report}
