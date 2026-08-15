"""Гвард измерений: не дать померить рычаг, который на самом деле не активен.

Корень двух ошибок сессии — замер без проверки, что рычаг включён в тестируемой
системе: флаг соседей не долетал до ретрива, разметка секций не долетала в
`Page.text`. Оба раза шум прогонов приписывался фиче (сравнение X с X). Здесь —
автоматические проверки, чтобы такой прогон ПАДАЛ, а не тихо мерил ерунду:

1. `graph_fingerprint` / `config_fingerprint` — слепок графа и конфига пишется в
   результат снимка; `kip_delta` потом проверяет, что before/after различаются
   ТОЛЬКО тестируемой осью (structural fingerprint графа обязан совпасть в
   retriever-only A/B — иначе граф пересобрали между прогонами, как в −10.7пп).
2. `preflight_neighbor_check` — перед прогоном убеждается, что `neighbors>0`
   реально добавляет соседние чанки на пробных вопросах (иначе abort).

Модуль намеренно без зависимостей от конкретного ретривера/коннектора — работает
с любым объектом, у которого есть `.run(cypher)` и `.retrieve(q)`.
"""

from __future__ import annotations


def graph_fingerprint(conn) -> dict:
    """Структурный слепок графа — то, что обязано быть ИДЕНТИЧНО в retriever-only A/B.

    Любое расхождение означает, что граф изменился между прогонами (пересборка,
    restore, недобитый билд) — и дельта уже не про тестируемый рычаг.
    """
    def one(q: str) -> int:
        return list(conn.run(q))[0]["x"]

    return {
        "nodes": one("MATCH (n) RETURN count(n) AS x"),
        "chunks_total": one("MATCH (n:Chunk) RETURN count(n) AS x"),
        "chunks_page": one(
            "MATCH (n:Chunk) WHERE n.id STARTS WITH 'chunk:page:' RETURN count(n) AS x"
        ),
        "kip_pages": one(
            "MATCH (p:Page) WHERE p.title STARTS WITH 'KIP-' RETURN count(p) AS x"
        ),
        # Сколько KIP-страниц реально размечено секциями (## ). Ключевой индикатор
        # того, ВКЛЮЧЁН ли section chunking: если ~2 из ~490 — секций нет, чанки блёклые.
        "kip_pages_sectioned": one(
            "MATCH (p:Page) WHERE p.title STARTS WITH 'KIP-' AND p.text CONTAINS '## ' "
            "RETURN count(p) AS x"
        ),
    }


def config_fingerprint(
    *, answer_mode: str, kip_reserve: int, kip_neighbors: int, top_k: int, rerank_top_k: int
) -> dict:
    """Слепок активного конфига ретрива/генерации, попадающий в результат снимка."""
    return {
        "answer_mode": answer_mode,
        "kip_reserve": int(kip_reserve),
        "kip_neighbors": int(kip_neighbors),
        "top_k": int(top_k),
        "rerank_top_k": int(rerank_top_k),
    }


def preflight_neighbor_check(retriever_with, retriever_without, probe_questions) -> int:
    """Убедиться, что neighbors реально добавляют кандидатов (иначе рычаг не активен).

    `retriever_with` — ретривер с текущим kip_neighbors (>0); `retriever_without` — с 0.
    Возвращает суммарное число добавленных соседних чанков по пробным вопросам.
    Падает AssertionError, если не добавилось НИЧЕГО — значит флаг не долетел до
    ретрива (ровно баг, обесценивший neighbor-A/B на −10.7пп).
    """
    added = 0
    for q in probe_questions:
        ids_w = {c.get("id") for c in retriever_with.retrieve(q).get("candidates", [])}
        ids_wo = {c.get("id") for c in retriever_without.retrieve(q).get("candidates", [])}
        added += len(ids_w - ids_wo)
    if added == 0:
        raise AssertionError(
            "PREFLIGHT: kip_neighbors>0, но соседние чанки не добавились ни на одном "
            "пробном вопросе — рычаг НЕ активен (флаг не долетел до ретрива?). "
            "Замер остановлен, чтобы не мерить X vs X."
        )
    return added


def diff_graph_fingerprints(fp_before: dict | None, fp_after: dict | None) -> list[str]:
    """Расхождения структурного слепка графа между before/after (пусто = графы совпали).

    Для retriever-only A/B (соседи, reserve, промпт) граф обязан быть идентичен.
    None (старый прогон без слепка) → возвращаем маркер, чтобы вызвавший предупредил,
    а не сделал вид, что всё проверено.
    """
    if fp_before is None or fp_after is None:
        return ["fingerprint отсутствует (старый прогон) — совпадение графа НЕ проверено"]
    diffs: list[str] = []
    for k in ("nodes", "chunks_total", "chunks_page", "kip_pages", "kip_pages_sectioned"):
        if fp_before.get(k) != fp_after.get(k):
            diffs.append(f"{k}: before={fp_before.get(k)} after={fp_after.get(k)}")
    return diffs


def diff_config_axes(cfg_before: dict | None, cfg_after: dict | None) -> list[str]:
    """Какие оси конфига различаются между before/after.

    Ожидание для чистого A/B — РОВНО одна ось. Пусто → это no-op (мерили одно и то же,
    как neighbor-A/B, где флаг не применялся). >1 → смешанный замер, эффект не изолирован.
    """
    if not cfg_before or not cfg_after:
        return []
    keys = set(cfg_before) | set(cfg_after)
    return [k for k in sorted(keys) if cfg_before.get(k) != cfg_after.get(k)]
