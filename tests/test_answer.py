"""Генерация с цитированием — на fake-LLM, без сети."""

from __future__ import annotations

from graphrag.generate.answer import (
    ANSWER_SYSTEM,
    ANSWER_SYSTEM_STRICT,
    ContextItem,
    build_context,
    extract_citations,
    generate_answer,
)
from graphrag.llm.base import LLMClient


class ScriptedLLM(LLMClient):
    def __init__(self, response: str):
        super().__init__("scripted")
        self.response = response
        self.calls = 0

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.calls += 1
        return self.response


class ExplodingLLM(LLMClient):
    def _raw_complete(self, *a, **kw):  # pragma: no cover
        raise AssertionError("LLM не должен вызываться при пустом контексте")


def test_extract_citations_dedup_and_order():
    text = "Факт A [источник: u1]. Факт B [источник: u2]. Ещё [источник: u1]."
    assert extract_citations(text) == ["u1", "u2"]


def test_build_context_from_chunks_and_impact():
    chunks = [{"text": "auth crashed", "uri": "u-chunk"}, {"text": "", "uri": "skip"}]
    impact = {
        "related_tasks": [{"key": "KAFKA-1", "summary": "fix", "status": "Resolved", "uri": "u-task"}],
        "related_pages": [{"title": "KIP", "uri": "u-page"}],
    }
    ctx = build_context(chunks, impact)
    uris = {it.uri for it in ctx}
    assert uris == {"u-chunk", "u-task", "u-page"}  # пустой чанк отброшен


def test_build_context_keeps_graph_candidate_with_uri():
    """Граф-кандидат с graph://-uri должен выживать в контексте (иначе multi-hop
    ask терял факты графа и отвечал «нет данных»)."""
    cands = [{"text": "Модуль connect связан с clients", "uri": "graph://module:connect"}]
    ctx = build_context(cands)
    assert len(ctx) == 1
    assert ctx[0].uri == "graph://module:connect"


def test_generate_with_context_has_valid_citation():
    ctx = [ContextItem(text="NetworkClient reconnect loop", uri="https://issues/KAFKA-101")]
    llm = ScriptedLLM("Проблема в клиенте [источник: https://issues/KAFKA-101].")
    res = generate_answer(llm, "что случилось?", ctx)
    assert res.grounded
    assert "https://issues/KAFKA-101" in res.citations
    assert llm.calls == 1


def test_empty_context_short_circuits_without_llm():
    """Пустой контекст -> ответ «нет данных», LLM не вызывается, фактов не выдумываем."""
    res = generate_answer(ExplodingLLM("x"), "вопрос", [])
    assert not res.grounded
    assert res.citations == []
    assert "недостаточно" in res.text.lower()


def test_hallucinated_citation_is_flagged_not_counted():
    """Цитата на uri вне контекста не засчитывается как валидная."""
    ctx = [ContextItem(text="реальный источник", uri="u-real")]
    llm = ScriptedLLM("Утверждение [источник: u-real] и выдумка [источник: u-fake].")
    res = generate_answer(llm, "q", ctx)
    assert res.citations == ["u-real"]
    assert res.hallucinated_citations == ["u-fake"]
    assert res.grounded  # есть хотя бы одна валидная


class CapturingLLM(LLMClient):
    """Фиксирует system-промпт, поданный в генерацию."""

    def __init__(self, response: str):
        super().__init__("capturing")
        self.response = response
        self.seen_system: str | None = None

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.seen_system = system
        return self.response


def test_strict_system_is_passed_through():
    """system=ANSWER_SYSTEM_STRICT доезжает до LLM (claim-conservative плечо A/B)."""
    ctx = [ContextItem(text="реальный источник", uri="u-real")]
    llm = CapturingLLM("Ответ [источник: u-real]")
    generate_answer(llm, "q", ctx, system=ANSWER_SYSTEM_STRICT)
    assert llm.seen_system == ANSWER_SYSTEM_STRICT


def test_default_system_unchanged():
    """Без system дефолт — прежний ANSWER_SYSTEM (поведение main не меняется)."""
    ctx = [ContextItem(text="реальный источник", uri="u-real")]
    llm = CapturingLLM("Ответ [источник: u-real]")
    generate_answer(llm, "q", ctx)
    assert llm.seen_system == ANSWER_SYSTEM


def test_strict_empty_context_still_short_circuits():
    """Строгий промпт не ломает инвариант: пустой контекст → LLM не зовётся."""
    res = generate_answer(ExplodingLLM("x"), "q", [], system=ANSWER_SYSTEM_STRICT)
    assert not res.grounded
    assert "недостаточно" in res.text.lower()
