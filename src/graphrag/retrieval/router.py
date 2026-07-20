"""Маршрутизатор интента запроса (Слой 3).

Факты → вектор+BM25; multi-hop/impact → обход графа; смешанные → комбинация.
Правила покрывают явные случаи; LLM (если дан) разрешает неоднозначные.
"""

from __future__ import annotations

from graphrag.llm.base import LLMClient

FACTUAL = "factual"
MULTIHOP = "multihop"
MIXED = "mixed"

_MULTIHOP_HINTS = (
    "сломает", "упад", "зависит", "зависят", "зависимост", "затрон", "影响",
    "impact", "влия", "каскад", "что будет если", "если откажет", "depends",
    "downstream", "обвал",
)
_FACTUAL_HINTS = (
    "что такое", "кто ", "когда ", "где описан", "определение", "как называется",
    "что делает", "для чего",
)

_ROUTE_PROMPT = (
    "Классифицируй вопрос по типу извлечения. Верни JSON {\"route\": ...} где route "
    'один из: "factual" (факт из документа), "multihop" (нужны связи/impact/'
    'зависимости), "mixed". Только JSON.\n\nВОПРОС: '
)


def classify_intent(
    question: str,
    llm: LLMClient | None = None,
    known_modules: list[str] | None = None,
) -> str:
    """Возвращает маршрут: factual | multihop | mixed.

    `MULTIHOP` (чистый обход графа) выбирается только когда есть и impact-подсказка,
    и упоминание известного модуля из `known_modules` — иначе impact-вопрос без
    имени модуля уходит в `MIXED` (вектор+BM25+граф), где граф пуст без модуля, но
    вектор находит контент. Без `known_modules` impact-подсказка одна не тянет в
    графовый путь: осмысленный обход всё равно требует имени модуля.
    """
    q = question.lower()
    if any(h in q for h in _MULTIHOP_HINTS):
        mods = known_modules or []
        if any(str(m).lower() in q for m in mods):
            return MULTIHOP
        return MIXED
    if any(h in q for h in _FACTUAL_HINTS):
        return FACTUAL

    if llm is not None:
        try:
            data = llm.extract_json(_ROUTE_PROMPT + question)
            route = str(data.get("route", "")).lower() if isinstance(data, dict) else ""
            if route in (FACTUAL, MULTIHOP, MIXED):
                return route
        except Exception:
            pass

    return MIXED
