"""Размеченный срез (чистый промпт + оркестратор), без живого Neo4j."""

from __future__ import annotations

import json

from eval.labeled_gen import (
    build_labeled_prompt,
    generate_labeled,
    generate_labeled_item,
)
from graphrag.llm.base import LLMClient


class ScriptedLLM(LLMClient):
    """Отдаёт валидную пару вопрос+эталон, уникализируя её по тексту источника."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        tail = prompt.strip()[-10:].replace('"', "'")
        return json.dumps(
            {"question": f"как чинить {tail}", "reference": f"эталон для {tail}"},
            ensure_ascii=False,
        )


class BadLLM(LLMClient):
    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        raise RuntimeError("boom")


# --- чистая промпт-функция ---

def test_prompt_demands_json_and_grounded_reference():
    prompt = build_labeled_prompt("NetworkClient reconnect loop")
    low = prompt.lower()
    assert "question" in low and "reference" in low  # требуемая форма JSON
    assert "только на текст" in low  # эталон опирается лишь на текст тикета
    assert "NetworkClient reconnect loop" in prompt


# --- generate_labeled_item ---

def test_item_happy():
    item = generate_labeled_item(ScriptedLLM("x"), "текст тикета про ребаланс")
    assert item is not None
    assert item["question"].strip() and item["reference"].strip()


def test_item_none_on_llm_failure():
    assert generate_labeled_item(BadLLM("x"), "текст") is None


def test_item_none_when_reference_missing():
    class HalfLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"question": "вопрос без эталона"}'

    assert generate_labeled_item(HalfLLM("x"), "текст") is None


# --- оркестратор ---

def test_generate_labeled_happy_and_serializable():
    sources = [
        {"source_id": "task:KAFKA-1", "text": "rebalance storm on restart"},
        {"source_id": "task:KAFKA-2", "text": "producer retries exhausted"},
    ]
    out = generate_labeled(ScriptedLLM("x"), sources, limit=40)
    assert len(out) == 2
    for rec, src in zip(out, sources):
        assert rec["question"].strip() and rec["reference"].strip()
        assert rec["source_id"] == src["source_id"]
    # Записи сериализуются в JSON (артефакт версионируется).
    assert json.loads(json.dumps(out, ensure_ascii=False)) == out


def test_generate_labeled_skips_empty_and_failures():
    sources = [
        {"source_id": "task:1", "text": "   "},
        {"source_id": "task:2", "text": "нормальный текст"},
    ]
    assert [r["source_id"] for r in generate_labeled(ScriptedLLM("x"), sources)] == ["task:2"]
    # Сбой на каждом элементе -> пусто, но без исключения.
    assert generate_labeled(BadLLM("x"), sources) == []


def test_generate_labeled_respects_limit():
    sources = [{"source_id": f"task:{i}", "text": f"уникальный текст {i}"} for i in range(10)]
    assert len(generate_labeled(ScriptedLLM("x"), sources, limit=3)) == 3
