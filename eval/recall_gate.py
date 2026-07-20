"""Recall-гейт: есть ли релевантный фрагмент в пуле кандидатов ДО реранка.

Проверяет гейтящее допущение плана — лимитирует ли качество контекста ранжирование
(нужный фрагмент в пуле, но плохо упорядочен) или поиск (фрагмента в пуле нет вовсе).
Ground truth — `source_id` вопроса (тикет/страница, из которого вопрос порождён).

Оговорка маршрута: MULTIHOP-вопросы объединяют только граф-кандидаты, поэтому
Task/Page-`source_id` там по построению даёт recall 0 — это оговорка при чтении, не баг.
"""

from __future__ import annotations

from eval.metrics import candidate_entity_id, recall_at_k


def pool_entity_ids(pool: list[dict]) -> list[str]:
    """id кандидатов пула, сведённые к entity-id (`chunk:task:K#0` -> `task:K`)."""
    return [candidate_entity_id(c["id"]) for c in pool]


def source_in_pool(pool: list[dict], source_id: str) -> float:
    """1.0, если `source_id` присутствует в пуле (recall чистого вхождения), иначе 0.0."""
    ids = pool_entity_ids(pool)
    return recall_at_k(ids, [source_id], k=len(ids)) if ids else 0.0


def evaluate_recall_gate(
    retriever, questions: list[dict], abstained: set[str]
) -> dict:
    """По каждому вопросу: попал ли `source_id` в пул кандидатов до реранка.

    `questions` — записи `{question, source_id}`; `abstained` — множество текстов
    вопросов, где система воздержалась в прошлом прогоне. Возвращает per-question
    записи + hit-rate со сплитом воздержавшиеся / отвечённые (go/no-go читается глазами).
    """
    records: list[dict] = []
    for q in questions:
        route, pool = retriever._candidate_pool(q["question"])
        records.append({
            "question": q["question"],
            "source_id": q["source_id"],
            "route": route,
            "pool_size": len(pool),
            "hit": source_in_pool(pool, q["source_id"]),
            "abstained": q["question"] in abstained,
        })

    def rate(subset: list[dict]) -> float:
        return sum(r["hit"] for r in subset) / len(subset) if subset else 0.0

    ab = [r for r in records if r["abstained"]]
    an = [r for r in records if not r["abstained"]]
    return {
        "records": records,
        "hit_rate_all": rate(records),
        "hit_rate_abstained": rate(ab),
        "hit_rate_answered": rate(an),
        "n_abstained": len(ab),
        "n_answered": len(an),
    }
