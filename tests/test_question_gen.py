"""Авто-генерация вопросов из корпуса (чистый промпт + оркестратор).

Без живого Neo4j: узлы-источники передаются в оркестратор напрямую. LLM —
скриптованный сабкласс LLMClient (как ScriptedLLM в test_metrics.py).
"""

from __future__ import annotations

from eval.question_gen import (
    build_question_prompt,
    generate_question,
    generate_questions,
)
from graphrag.llm.base import LLMClient


class ScriptedLLM(LLMClient):
    """Отдаёт валидный JSON-вопрос, слегка меняя его под текст источника."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        # Уникализируем вопрос по хвосту промпта, чтобы дедуп не схлопывал набор.
        tail = prompt.strip()[-12:].replace('"', "'")
        return f'{{"question": "как обойти проблему {tail}"}}'


class BadLLM(LLMClient):
    """Всегда бросает — эмуляция сбоя судьи/генератора на элементе."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        raise RuntimeError("boom")


# --- чистая промпт-функция ---

def test_prompt_contains_anti_trivial_guidance():
    prompt = build_question_prompt("clients reconnect loop under outage")
    low = prompt.lower()
    # Уклон в сквозные формулировки и явный запрет тривиального мета-вопроса.
    assert "почему" in low
    assert "затрон" in low
    assert "о чём" in low  # упомянут как запрещённый шаблон
    # Текст источника попал в промпт.
    assert "clients reconnect loop" in prompt


# --- generate_question (тонкая обёртка) ---

def test_generate_question_happy():
    q = generate_question(ScriptedLLM("x"), "какой-то текст тикета")
    assert isinstance(q, str) and q.strip()


def test_generate_question_none_on_failure():
    assert generate_question(BadLLM("x"), "текст") is None


def test_generate_question_none_on_empty_answer():
    class EmptyLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"question": "   "}'

    assert generate_question(EmptyLLM("x"), "текст") is None


# --- оркестратор ---

def test_generate_questions_happy():
    sources = [
        {"source_id": "task:1", "text": "clients reconnect loop under outage"},
        {"source_id": "task:2", "text": "producer timeout on send retries"},
        {"source_id": "page:1", "text": "как настроить репликацию брокеров"},
    ]
    out = generate_questions(ScriptedLLM("x"), sources, limit=200)
    assert len(out) == 3
    for rec, src in zip(out, sources):
        assert rec["question"].strip()
        assert rec["source_id"] == src["source_id"]


def test_generate_questions_skips_empty_text():
    sources = [
        {"source_id": "task:1", "text": "  "},
        {"source_id": "task:2", "text": ""},
        {"source_id": "task:3", "text": "реальный текст тикета"},
    ]
    out = generate_questions(ScriptedLLM("x"), sources)
    assert [r["source_id"] for r in out] == ["task:3"]


def test_generate_questions_skips_llm_failure_without_crash():
    # Генератор падает на КАЖДОМ элементе -> пустой набор, но без исключения.
    sources = [
        {"source_id": "task:1", "text": "текст один"},
        {"source_id": "task:2", "text": "текст два"},
    ]
    out = generate_questions(BadLLM("x"), sources)
    assert out == []


def test_generate_questions_dedups_and_limits():
    class ConstLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"question": "один и тот же вопрос"}'

    sources = [{"source_id": f"task:{i}", "text": f"текст {i}"} for i in range(5)]
    # Дедуп: пять одинаковых вопросов -> одна запись.
    out = generate_questions(ConstLLM("x"), sources)
    assert len(out) == 1

    # Лимит режет набор.
    limited = generate_questions(
        ScriptedLLM("x"),
        [{"source_id": f"t:{i}", "text": f"уникальный текст номер {i}"} for i in range(10)],
        limit=3,
    )
    assert len(limited) == 3
