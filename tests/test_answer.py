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


# --- claim-conservative промпт + system-override ---

class RecordingSystemLLM(LLMClient):
    def __init__(self):
        super().__init__("rec")
        self.systems: list = []

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.systems.append(system)
        return "ответ [источник: u-real]"


def test_default_system_is_answer_system():
    llm = RecordingSystemLLM()
    generate_answer(llm, "q", [ContextItem(text="ctx", uri="u-real")])
    assert llm.systems == [ANSWER_SYSTEM]  # дефолт — текущий промпт (прод не меняется)


def test_system_override_passed_through():
    llm = RecordingSystemLLM()
    generate_answer(llm, "q", [ContextItem(text="ctx", uri="u-real")], system=ANSWER_SYSTEM_STRICT)
    assert llm.systems == [ANSWER_SYSTEM_STRICT]  # A/B-плечо получает строгий промпт


def test_strict_prompt_forbids_overreach():
    low = ANSWER_SYSTEM_STRICT.lower()
    assert "только" in low  # утверждать только влекомое
    assert "не указано" in low or "не сказано" in low  # явно помечать отсутствующее
    assert "источник" in low  # цитаты сохранены


def test_hallucinated_citation_is_flagged_not_counted():
    """Цитата на uri вне контекста не засчитывается как валидная."""
    ctx = [ContextItem(text="реальный источник", uri="u-real")]
    llm = ScriptedLLM("Утверждение [источник: u-real] и выдумка [источник: u-fake].")
    res = generate_answer(llm, "q", ctx)
    assert res.citations == ["u-real"]
    assert res.hallucinated_citations == ["u-fake"]
    assert res.grounded  # есть хотя бы одна валидная
