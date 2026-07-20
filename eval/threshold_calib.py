"""Калибровка порога отсечения слабых фрагментов с guardrail на воздержания.

По распределению cross-encoder-скоров выбираем кандидатные пороги, перепрогоняем с
каждым и оставляем порог, только если он поднимает context precision БЕЗ роста
воздержаний относительно cross-encoder без порога (R4). Иначе порог остаётся 0.

Ограничение шкалы: `filter_by_threshold` выключается при `min_score <= 0`, а
cross-encoder-скоры — логиты (могут быть <= 0). Поэтому кандидатные пороги ищем
строго в области `> 0`; если разделяющая граница <= 0 — порог невыразим при текущей
семантике фильтра (кандидатов нет).
"""

from __future__ import annotations


def positive_scores(scores: list[float]) -> list[float]:
    """Только строго положительные логиты (выразимые как порог)."""
    return [s for s in scores if s > 0]


def candidate_thresholds(scores: list[float]) -> list[float]:
    """Кандидатные пороги — квартильные точки среди положительных скоров.

    Пусто, если положительных скоров нет (порог невыразим — см. модульный докстринг).
    """
    pos = sorted(positive_scores(scores))
    if not pos:
        return []
    n = len(pos)
    out = {pos[min(n - 1, int(q * n))] for q in (0.25, 0.5, 0.75)}
    return sorted(out)


def recommend_threshold(baseline: dict, evals: list[dict]) -> dict:
    """Выбор порога с guardrail: precision вверх И воздержания не выше базовой линии.

    `baseline` / элементы `evals` несут `precision` и `abstention`; `evals` — ещё
    `threshold`. Возвращает `{threshold, on, reason}`; при отсутствии подходящего —
    порог 0 (выключен).
    """
    qualifying = [
        e for e in evals
        if e["precision"] > baseline["precision"]
        and e["abstention"] <= baseline["abstention"]
    ]
    if not qualifying:
        return {
            "threshold": 0.0,
            "on": False,
            "reason": "ни один порог не поднял precision без роста воздержаний",
        }
    best = max(qualifying, key=lambda e: e["precision"])
    return {
        "threshold": best["threshold"],
        "on": True,
        "reason": (
            f"precision {baseline['precision']:.2f}->{best['precision']:.2f}, "
            f"воздержания {baseline['abstention']:.2f}->{best['abstention']:.2f}"
        ),
    }


T = "eval/trial"


def main() -> None:
    """Онлайн-калибровка (нужны --extra ml + Neo4j + LLM).

    Скоры собираются из `retriever.retrieve(q)["candidates"]` (там есть `rerank_score`;
    `evaluate_question` его НЕ отдаёт), граф-кандидаты исключаются (порог их не трогает).
    """
    import json

    import torch

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder
    from graphrag.embeddings.reranker import CrossEncoderReranker
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm

    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.quality_eval import evaluate_question

    torch.set_num_threads(2)
    s = load_settings()
    questions = json.load(open(f"{T}/questions_real.json", encoding="utf-8"))
    labeled = json.load(open(f"{T}/labeled_real.json", encoding="utf-8"))
    items = questions + labeled

    def build(min_score: float) -> HybridRetriever:
        return HybridRetriever(
            conn, emb, CrossEncoderReranker(s.reranker.model),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=min_score,
        )

    def measure(retr) -> dict:
        recs = [evaluate_question(retr, llm, it["question"],
                                  reference=it.get("reference"),
                                  source_id=it.get("source_id")) for it in items]
        precs = [r["metrics"]["context_precision"] for r in recs
                 if r["metrics"]["context_precision"] is not None]
        ab = sum(1 for r in recs if r["abstained"]["faithfulness"])
        return {"precision": sum(precs) / len(precs) if precs else 0.0,
                "abstention": ab / len(recs) if recs else 0.0}

    with Neo4jConnection(s.neo4j) as conn:
        emb = build_embedder(s.embeddings)
        llm = build_llm(s.llm, role="generation")

        # собрать скоры (без графа — он освобождён от порога)
        retr0 = build(0.0)
        scores: list[float] = []
        for it in items:
            for c in retr0.retrieve(it["question"])["candidates"]:
                if c.get("source") != "graph" and "rerank_score" in c:
                    scores.append(c["rerank_score"])

        baseline = measure(retr0)
        cands = candidate_thresholds(scores)
        evals = [{"threshold": t, **measure(build(t))} for t in cands]

    rec = recommend_threshold(baseline, evals)
    out = {"baseline": baseline, "candidates": evals, "recommendation": rec,
           "n_scores": len(scores)}
    json.dump(out, open(f"{T}/threshold_calib.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # ASCII-краткая сводка (полное — в utf-8-файле; Windows-консоль cp1251 падает на не-ASCII)
    print(f"baseline precision={baseline['precision']:.2f} abstention={baseline['abstention']:.2f}")
    print(f"candidates={len(evals)} scores={len(scores)} -> on={rec['on']} threshold={rec['threshold']}")
    print("DONE -> eval/trial/threshold_calib.json")


if __name__ == "__main__":
    main()
