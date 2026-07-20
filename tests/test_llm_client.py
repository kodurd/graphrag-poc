"""Контракт LLMClient и устойчивый разбор JSON — без сети."""

from __future__ import annotations

import pytest

from graphrag.llm.base import LLMClient, LLMError, parse_json


class ScriptedLLM(LLMClient):
    """Fake-клиент: отдаёт заранее заданные ответы по очереди."""

    def __init__(self, responses: list[str], **kw):
        super().__init__("scripted", **kw)
        self._responses = list(responses)
        self.calls = 0

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.calls += 1
        return self._responses.pop(0)


# --- parse_json (чистая функция) ---

def test_parse_json_plain_object():
    assert parse_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_fenced():
    raw = "Вот результат:\n```json\n{\"nodes\": [\"auth\"]}\n```\nготово"
    assert parse_json(raw) == {"nodes": ["auth"]}


def test_parse_json_surrounding_text():
    raw = 'бла-бла [1, 2, 3] хвост'
    assert parse_json(raw) == [1, 2, 3]


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError):
        parse_json("совсем не json")


# --- extract_json (happy) ---

def test_extract_json_happy_schema():
    """Модель вернула валидный JSON извлечения сущностей — разбираем."""
    client = ScriptedLLM(['{"services": ["auth-service"], "exceptions": ["NPE"]}'])
    result = client.extract_json("извлеки сущности из лога")
    assert result == {"services": ["auth-service"], "exceptions": ["NPE"]}
    assert client.calls == 1


# --- extract_json (ретрай на мусоре) ---

def test_extract_json_retries_then_succeeds():
    """Первый ответ — мусор, второй — валидный JSON. Должен подхватить второй."""
    client = ScriptedLLM(["не json вообще", '{"ok": true}'], max_retries=2)
    result = client.extract_json("верни JSON")
    assert result == {"ok": True}
    assert client.calls == 2


def test_extract_json_all_invalid_raises_not_crash():
    """Всегда мусор → аккуратная LLMError, а не необработанное падение."""
    client = ScriptedLLM(["мусор1", "мусор2", "мусор3"], max_retries=2)
    with pytest.raises(LLMError):
        client.extract_json("верни JSON")
    assert client.calls == 3  # 1 + max_retries


# --- complete (простое прохождение) ---

def test_complete_passthrough():
    client = ScriptedLLM(["ответ модели"])
    assert client.complete("вопрос") == "ответ модели"
