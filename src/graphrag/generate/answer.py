"""Генерация ответа с обязательным цитированием.

В госухе ответам без ссылок не доверяют: контекст собирается с uri источников,
промпт требует цитаты вида [источник: uri], а пост-проверка гарантирует, что
цитаты ссылаются только на реально поданные источники (без галлюцинаций ссылок).
Пустой контекст → честное «недостаточно данных», без выдуманных фактов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphrag.llm.base import LLMClient

ANSWER_SYSTEM = (
    "Ты отвечаешь строго на основе поданного контекста. К каждому утверждению "
    "добавляй ссылку на источник в формате [источник: <uri>]. Если данных "
    "недостаточно — прямо скажи об этом и не выдумывай."
)

# Claim-conservative вариант против переобобщения: на снимке n=96 главный полюс низкой
# faithfulness — генератор дописывал неподкреплённое («также затронуто X», «полностью
# решит Y», граф-домыслы) на ХОРОШЕМ контексте. Явно запрещает выводы за пределами
# контекста и требует помечать отсутствующее.
ANSWER_SYSTEM_STRICT = (
    "Ты отвечаешь СТРОГО на основе поданного контекста. Правила:\n"
    "1) Утверждай ТОЛЬКО то, что прямо следует из контекста. Никаких выводов, обобщений "
    "или связей, которых в контексте нет.\n"
    "2) НЕ добавляй «также затронуто X», «повлияет на Y», «полностью решит Z» и подобное, "
    "если этого нет в контексте.\n"
    "3) Если чего-то в контексте нет — прямо скажи «в источнике не указано», а не додумывай.\n"
    "4) К каждому утверждению — ссылка [источник: <uri>]. Недостаточно данных — так и скажи, "
    "не выдумывай."
)

# Partial-вариант против ПЕРЕ-воздержания: диагностик отказов показал — у 80% отказов
# контекст релевантен (15% ответ прямо есть, 67% частичный), но faith-first генератор
# отказывается полностью. Этот промпт разрешает частичный ОБОСНОВАННЫЙ ответ + явную
# пометку пробелов, оставляя запрет на выдумки. A/B со стоп-правилом по faithfulness.
ANSWER_SYSTEM_PARTIAL = (
    "Ты отвечаешь на основе поданного контекста. Правила:\n"
    "1) Если контекст покрывает вопрос ХОТЯ БЫ ЧАСТИЧНО — ответь на покрытую часть и ЯВНО "
    "отметь, чего в источнике нет («по остальному в источнике не указано»). НЕ отказывайся "
    "полностью, когда релевантная информация есть.\n"
    "2) Утверждай ТОЛЬКО то, что следует из контекста; не выдумывай факты и связи, которых "
    "в нём нет.\n"
    "3) К каждому утверждению — ссылка [источник: <uri>].\n"
    "4) Полный отказ «недостаточно данных» — ТОЛЬКО если контекст вообще не относится к вопросу."
)

# Выбор системного промпта по режиму (для A/B через env ANSWER_MODE).
ANSWER_SYSTEMS = {
    "default": ANSWER_SYSTEM,
    "strict": ANSWER_SYSTEM_STRICT,
    "partial": ANSWER_SYSTEM_PARTIAL,
}


def system_for_mode(mode: str | None) -> str:
    """Системный промпт по имени режима; неизвестный/пустой → default."""
    return ANSWER_SYSTEMS.get((mode or "default").strip().lower(), ANSWER_SYSTEM)


_CITATION_RE = re.compile(r"\[источник:\s*([^\]]+)\]")


@dataclass
class ContextItem:
    text: str
    uri: str
    kind: str = "chunk"


@dataclass
class AnswerResult:
    text: str
    citations: list[str]  # валидные (присутствуют в контексте)
    grounded: bool
    hallucinated_citations: list[str] = field(default_factory=list)


def build_context(
    chunks: list[dict] | None = None, impact: dict | None = None
) -> list[ContextItem]:
    """Собирает контекст из чанков (вектор/BM25) и фактов impact-подграфа."""
    items: list[ContextItem] = []

    for ch in chunks or []:
        uri = ch.get("uri") or ""
        text = ch.get("text") or ""
        if text and uri:
            items.append(ContextItem(text=text, uri=uri, kind="chunk"))

    if impact:
        for t in impact.get("related_tasks", []):
            if t.get("uri"):
                items.append(
                    ContextItem(
                        text=f"Тикет {t.get('key')}: {t.get('summary')} [{t.get('status')}]",
                        uri=t["uri"],
                        kind="task",
                    )
                )
        for p in impact.get("related_pages", []):
            if p.get("uri"):
                items.append(ContextItem(text=f"Страница: {p.get('title')}", uri=p["uri"], kind="page"))

    return items


MAX_CONTEXT_ITEM_CHARS = 1500


def build_prompt(question: str, context: list[ContextItem]) -> str:
    lines = [f"Вопрос: {question}", "", "Контекст (используй только его):"]
    for i, it in enumerate(context, 1):
        # Ограничиваем длину фрагмента: корпусный текст (вики/тикеты) недоверенный,
        # длинный фрагмент — лишняя поверхность prompt-injection и раздутый промпт.
        text = it.text[:MAX_CONTEXT_ITEM_CHARS]
        lines.append(f"[{i}] (источник: {it.uri}) {text}")
    lines += ["", "Ответь по существу, сопровождая утверждения цитатами [источник: <uri>]."]
    return "\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """Достаёт uri из цитат [источник: uri] в порядке появления, без дублей."""
    seen: list[str] = []
    for m in _CITATION_RE.findall(text):
        uri = m.strip()
        if uri not in seen:
            seen.append(uri)
    return seen


def generate_answer(
    llm: LLMClient, question: str, context: list[ContextItem], *, system: str = ANSWER_SYSTEM
) -> AnswerResult:
    """Генерирует ответ; при пустом контексте не зовёт LLM и не выдумывает.

    `system` переключает промпт (дефолт — текущий; `ANSWER_SYSTEM_STRICT` —
    claim-conservative для A/B против переобобщения).
    """
    if not context:
        return AnswerResult(
            text="Недостаточно данных в базе, чтобы ответить на этот вопрос.",
            citations=[],
            grounded=False,
        )

    raw = llm.complete(build_prompt(question, context), system=system)
    cited = extract_citations(raw)
    valid_uris = {it.uri for it in context}
    valid = [c for c in cited if c in valid_uris]
    hallucinated = [c for c in cited if c not in valid_uris]

    return AnswerResult(
        text=raw,
        citations=valid,
        grounded=len(valid) > 0,
        hallucinated_citations=hallucinated,
    )
