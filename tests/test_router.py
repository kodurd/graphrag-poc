"""Маршрутизатор интента — чистая логика."""

from __future__ import annotations

from graphrag.llm.base import LLMClient
from graphrag.retrieval.router import FACTUAL, MIXED, MULTIHOP, classify_intent


MODS = ["clients", "connect", "streams"]


def test_multihop_needs_hint_and_named_module():
    # impact-подсказка И имя известного модуля -> чистый графовый путь
    assert classify_intent("что сломается, если упадёт clients?", known_modules=MODS) == MULTIHOP
    assert classify_intent("какие модули зависят от connect", known_modules=MODS) == MULTIHOP


def test_impact_hint_without_module_goes_mixed():
    # impact-подсказка есть, но модуль не назван -> MIXED (граф пуст, вектор найдёт)
    assert classify_intent("что сломает изменение RPC-контракта", known_modules=MODS) == MIXED
    # нет списка модулей -> одна подсказка не тянет в графовый путь
    assert classify_intent("что затронет это изменение") == MIXED
    assert classify_intent("что затронет это изменение", known_modules=[]) == MIXED


def test_factual_by_keyword():
    assert classify_intent("что такое NetworkClient") == FACTUAL
    assert classify_intent("кто отвечает за clients") == FACTUAL


def test_mixed_default_without_llm():
    assert classify_intent("расскажи про kafka connect") == MIXED


def test_llm_resolves_ambiguous():
    class ScriptedLLM(LLMClient):
        def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
            return '{"route": "multihop"}'

    # нет ключевых слов -> обычно mixed, но LLM переводит в multihop
    assert classify_intent("а что потом посыпется", ScriptedLLM("x")) == MULTIHOP
