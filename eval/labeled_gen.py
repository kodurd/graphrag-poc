"""Генерация размеченного среза: тикет -> {вопрос, эталонный ответ}.

Маленький срез (десятки) для метрик, которым нужна истина — answer correctness и
context recall. Эталон порождает LLM по тексту тикета, то есть **эталон сам
LLM-generated и может врать**; это учитывается при интерпретации метрик, а не
выдаётся за абсолютную истину.

Чистая функция построения промпта (`build_labeled_prompt`) отделена от тонкой
обёртки вызова (`generate_labeled_item`), чтобы тестировать без сети. Обход графа
повторяет `golden_set.build_from_graph`.
"""

from __future__ import annotations

from graphrag.graph.connection import Neo4jConnection
from graphrag.llm.base import LLMClient

_LABELED_PROMPT = (
    "По тексту тикета сформулируй ОДИН реалистичный вопрос и эталонный ответ на "
    "него.\n"
    "Вопрос — такой, какой инженер задал бы поисковой системе; самодостаточный "
    "(не ссылайся на «этот тикет» и не подставляй его идентификатор).\n"
    "Эталонный ответ — краткий и опирающийся ТОЛЬКО на текст тикета; не добавляй "
    "фактов, которых в тексте нет.\n"
    'Верни строго JSON {"question": "<вопрос>", "reference": "<эталон>"}. '
    "Только JSON."
)


def build_labeled_prompt(source_text: str) -> str:
    """Строит промпт генерации пары «вопрос + эталон» (чистая функция)."""
    return f"{_LABELED_PROMPT}\n\nТЕКСТ ТИКЕТА:\n{source_text.strip()}"


def generate_labeled_item(llm: LLMClient, source_text: str) -> dict | None:
    """Тонкая обёртка: текст тикета -> {question, reference} либо None при сбое.

    None при сбое (сеть/невалидный JSON/пустое поле), чтобы сбой на одном элементе
    не ронял прогон — по образцу `judge_faithfulness`.
    """
    try:
        data = llm.extract_json(build_labeled_prompt(source_text))
        if not isinstance(data, dict):
            return None
        question = str(data.get("question") or "").strip()
        reference = str(data.get("reference") or "").strip()
        if not question or not reference:
            return None
        return {"question": question, "reference": reference}
    except Exception:
        return None


def collect_ticket_sources(conn: Neo4jConnection) -> list[dict]:
    """Собирает {source_id, text} из Task с непустым текстом (summary+description)."""
    rows = conn.run(
        "MATCH (n:Task) RETURN n.id AS id, "
        "coalesce(n.summary,'') + '\\n' + coalesce(n.description,'') AS text"
    )
    out: list[dict] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if text:
            out.append({"source_id": r["id"], "text": text})
    return out


def generate_labeled(
    llm: LLMClient, sources: list[dict], *, limit: int = 40
) -> list[dict]:
    """Оркестратор: источники -> записи {question, reference, source_id}.

    Источники передаются напрямую (тестируется без Neo4j). Узлы без текста
    пропускаются; сбой LLM на элементе пропускает элемент, не роняя прогон.
    """
    out: list[dict] = []
    for src in sources:
        if len(out) >= limit:
            break
        text = str(src.get("text") or "").strip()
        if not text:
            continue
        item = generate_labeled_item(llm, text)
        if not item:
            continue
        out.append({**item, "source_id": src.get("source_id")})
    return out


def generate_from_graph(
    conn: Neo4jConnection, llm: LLMClient, *, limit: int = 40
) -> list[dict]:
    """Сквозной вход: собрать тикеты из графа и породить размеченный срез."""
    return generate_labeled(llm, collect_ticket_sources(conn), limit=limit)
