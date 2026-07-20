"""Авто-генерация вопросов из корпуса: тикеты/страницы -> реалистичные вопросы.

Для каждого текстонесущего узла (Task/Page с непустым текстом) DeepSeek порождает
один нетривиальный вопрос — уклон в «как решить / почему / что затронуто», а не
«о чём тикет X». Чистая функция построения промпта (`build_question_prompt`)
отделена от тонкой обёртки вызова LLM (`generate_question`), чтобы промпт можно
было тестировать без сети. Обход графа повторяет `golden_set.build_from_graph`.
"""

from __future__ import annotations

from graphrag.graph.connection import Neo4jConnection
from graphrag.llm.base import LLMClient

# Какие узлы дают текст-источник (label, cypher-выражение текста) — как TEXT_SOURCES
# в индексаторе, но только реалистичные носители пользовательского смысла.
TEXT_SOURCES: list[tuple[str, str]] = [
    ("Task", "coalesce(n.summary,'') + '\\n' + coalesce(n.description,'')"),
    ("Page", "coalesce(n.title,'') + '\\n' + coalesce(n.text,'')"),
]

# Анти-тривиальные указания вынесены в константу: на них опирается тест промпта и
# от них зависит уклон набора в сквозные/реалистичные формулировки.
_ANTI_TRIVIAL = (
    "Сформулируй ОДИН реалистичный, нетривиальный вопрос, который живой инженер "
    "задал бы поисковой RAG-системе, опираясь на этот текст как на контекст.\n"
    "Уклон в сквозные формулировки: «как решить/обойти», «почему так происходит», "
    "«что будет затронуто/сломается», «как связано с другими тикетами и модулями».\n"
    "ЗАПРЕЩЕНЫ тривиальные мета-вопросы вида «о чём этот тикет», «как называется "
    "страница», «какой у тикета статус» — они не проверяют качество ответа.\n"
    "Вопрос должен быть самодостаточным: не ссылайся на «этот тикет/эту страницу» "
    "и не подставляй его идентификатор — читатель вопроса текста не видит.\n"
    'Верни строго JSON {"question": "<текст вопроса>"}. Только JSON.'
)


def build_question_prompt(source_text: str) -> str:
    """Строит промпт генерации вопроса для текста источника (чистая функция).

    Тестируется без сети: содержит анти-тривиальные указания (`_ANTI_TRIVIAL`) и
    сам текст источника.
    """
    return f"{_ANTI_TRIVIAL}\n\nТЕКСТ ИСТОЧНИКА:\n{source_text.strip()}"


def generate_question(llm: LLMClient, source_text: str) -> str | None:
    """Тонкая обёртка вызова LLM: текст источника -> вопрос или None при сбое.

    Возвращает None при сбое (сеть/невалидный JSON/пустой вопрос), чтобы сбой на
    одном элементе не ронял весь прогон — по образцу `judge_faithfulness`.
    """
    prompt = build_question_prompt(source_text)
    try:
        data = llm.extract_json(prompt)
        if not isinstance(data, dict) or "question" not in data:
            return None
        question = str(data["question"]).strip()
        return question or None
    except Exception:
        return None


def collect_text_sources(
    conn: Neo4jConnection, labels: list[str] | None = None
) -> list[dict]:
    """Собирает источники {source_id, text} из Task/Page с непустым текстом.

    Обход графа — как в `golden_set.build_from_graph`: conn.run -> список dict'ов.
    Узлы без текста отсеиваются здесь же.
    """
    out: list[dict] = []
    for label, text_expr in TEXT_SOURCES:
        if labels and label not in labels:
            continue
        rows = conn.run(f"MATCH (n:{label}) RETURN n.id AS id, {text_expr} AS text")
        for r in rows:
            text = (r.get("text") or "").strip()
            if text:
                out.append({"source_id": r["id"], "text": text})
    return out


def _normalize(question: str) -> str:
    """Ключ дедупликации: без регистра и краевых пробелов."""
    return question.strip().lower()


def generate_questions(
    llm: LLMClient, sources: list[dict], *, limit: int = 200
) -> list[dict]:
    """Оркестратор: по источникам порождает вопросы {question, source_id}.

    Источники — список dict'ов с ключами `source_id` и `text` (передаётся напрямую,
    поэтому тестируется без Neo4j). Узлы без текста пропускаются; сбой LLM на
    элементе (generate_question -> None) пропускает элемент, не роняя прогон.
    Результаты дедуплицируются по тексту вопроса и обрезаются до `limit`.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for src in sources:
        if len(out) >= limit:
            break
        text = str(src.get("text") or "").strip()
        if not text:
            continue
        question = generate_question(llm, text)
        if not question:
            continue
        key = _normalize(question)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": question, "source_id": src.get("source_id")})
    return out


def generate_from_graph(
    conn: Neo4jConnection,
    llm: LLMClient,
    *,
    limit: int = 200,
    labels: list[str] | None = None,
) -> list[dict]:
    """Сквозной вход: собрать источники из графа и породить набор вопросов."""
    sources = collect_text_sources(conn, labels)
    return generate_questions(llm, sources, limit=limit)
