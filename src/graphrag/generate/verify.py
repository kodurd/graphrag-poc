"""verify-then-answer: механическая пост-фильтрация переобобщения.

Диагноз: генератор берёт обоснованное ядро из контекста и дописывает сверху
неподтверждённое (faithfulness падает). Промптовый рычаг «не выдумывай» доказанно не
работает. Здесь неподтверждённое убирается ДЕЙСТВИЕМ: ответ раскладывается на утверждения,
каждое сверяется с контекстом, итог пересобирается только из подтверждённых (поданных как
единственный источник), с одной повторной проверкой; при сбое/малом ядре/остаточном домысле —
честное воздержание (fail-closed).

Декомпозер держим здесь (в src), а не импортируем из eval/claim_survival, чтобы прод-код не
зависел от eval-пакета. Промпт расширен: по подтверждённому утверждению возвращается uri
поддерживающего фрагмента — для сохранения цитат при регенерации.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from graphrag.generate.answer import ContextItem, generate_answer
from graphrag.llm.base import LLMClient

# Предрегистрированные пороги (тюнятся в A/B, но заданы явно).
MIN_SUPPORTED = 2  # минимум подтверждённых утверждений для регенерации, иначе воздержание
RESIDUAL_THRESHOLD = 0.2  # доля остаточного домысла после регенерации выше → воздержание

_DECOMPOSE_PROMPT = (
    "Дан ОТВЕТ и пронумерованные фрагменты КОНТЕКСТА (каждый с источником [источник: uri]). "
    "Разложи ОТВЕТ на атомарные фактические утверждения. Для каждого:\n"
    "- supported=true, если оно ПРЯМО следует из какого-то фрагмента — тогда в поле source "
    "укажи uri этого фрагмента;\n"
    "- supported=false, если это домысел/обобщение/связь, которой в контексте нет (source не нужен).\n"
    "Оговорки, вопросы и фразы «в источнике не указано» — не утверждения, пропусти. "
    'Верни ТОЛЬКО JSON: {"claims": [{"claim": "...", "supported": true, "source": "<uri>"}]}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class VerifyResult:
    text: str
    abstained: bool
    abstain_reason: str | None  # parse_fail | too_few | empty_regen | residual | None
    citations: list[str]
    grounded: bool
    n_claims: int
    n_supported: int
    residual_frac: float | None


def parse_claims(raw: str) -> list[dict] | None:
    """Парсит ответ декомпозера → список {claim, supported, source} либо None при мусоре."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    claims = data.get("claims")
    if not isinstance(claims, list):
        return None
    out: list[dict] = []
    for c in claims:
        if not isinstance(c, dict) or "claim" not in c:
            continue
        out.append({
            "claim": str(c.get("claim", "")),
            "supported": c.get("supported") is True,
            "source": str(c.get("source") or "") or None,
        })
    return out


def supported_items(claims: list[dict]) -> list[dict]:
    """Подтверждённые утверждения (текст + source-uri)."""
    return [c for c in claims if c.get("supported")]


def _decompose(llm: LLMClient, question: str, answer: str, context: list[ContextItem]) -> list[dict] | None:
    """Разложить ОТВЕТ на утверждения и сверить с контекстом (может вернуть None при сбое)."""
    ctx_block = "\n".join(f"[источник: {it.uri}] {it.text}" for it in context)
    prompt = f"{_DECOMPOSE_PROMPT}\n\nКОНТЕКСТ:\n{ctx_block}\n\nОТВЕТ:\n{answer}"
    try:
        return parse_claims(llm.complete(prompt))
    except Exception:  # noqa: BLE001 — сбой декомпозера трактуем как отсутствие данных
        return None


def _abstain(reason: str, claims: list[dict] | None = None, supported: list[dict] | None = None) -> VerifyResult:
    return VerifyResult(
        text="Недостаточно подтверждённых данных, чтобы ответить без домыслов.",
        abstained=True, abstain_reason=reason, citations=[], grounded=False,
        n_claims=len(claims or []), n_supported=len(supported or []), residual_frac=None,
    )


def verify_then_answer(
    llm: LLMClient,
    question: str,
    answer: str,
    context: list[ContextItem],
    *,
    min_supported: int = MIN_SUPPORTED,
    residual_threshold: float = RESIDUAL_THRESHOLD,
) -> VerifyResult:
    """Отфильтровать переобобщение: пересобрать ответ из подтверждённых утверждений.

    Ветки воздержания (fail-closed): сбой декомпозера (parse_fail), мало подтверждённых
    (too_few), пустая регенерация (empty_regen), остаточный домысел выше порога (residual).
    """
    claims = _decompose(llm, question, answer, context)
    if claims is None:
        return _abstain("parse_fail")

    supported = supported_items(claims)
    if len(supported) < min_supported:
        return _abstain("too_few", claims, supported)

    # Регенерация: подтверждённые утверждения — единственный источник, с их uri (цитаты живут).
    regen_ctx = [ContextItem(text=c["claim"], uri=c["source"] or "") for c in supported if c["source"]]
    if not regen_ctx:  # подтверждённые без uri — цитировать нечем
        return _abstain("too_few", claims, supported)
    regen = generate_answer(llm, question, regen_ctx)
    if not regen.text.strip() or not regen.grounded:
        return _abstain("empty_regen", claims, supported)

    # Пере-суд один раз: доля остаточного домысла в пересобранном ответе.
    recheck = _decompose(llm, question, regen.text, regen_ctx)
    residual_frac = 0.0
    if recheck:
        n_unsup = sum(1 for c in recheck if not c.get("supported"))
        residual_frac = n_unsup / len(recheck) if recheck else 0.0
    if residual_frac > residual_threshold:
        return _abstain("residual", claims, supported)

    return VerifyResult(
        text=regen.text, abstained=False, abstain_reason=None,
        citations=regen.citations, grounded=regen.grounded,
        n_claims=len(claims), n_supported=len(supported), residual_frac=residual_frac,
    )
