"""Базовый интерфейс LLM-клиента + устойчивый разбор JSON.

Реализации (api.py, ollama.py) переопределяют только `_raw_complete`.
`complete` и `extract_json` — общие, с ретраями на невалидном JSON.
"""

from __future__ import annotations

import abc
import json
import re

JSON_SYSTEM = (
    "Ты извлекаешь данные и отвечаешь ТОЛЬКО валидным JSON без пояснений "
    "и без markdown-обёрток."
)


class LLMError(RuntimeError):
    """Ошибка вызова модели или разбора её ответа."""


def parse_json(raw: str) -> dict | list:
    """Достаёт JSON из ответа модели.

    Терпит ```json-ограждения и текст вокруг: берёт первый сбалансированный
    объект/массив. Бросает ValueError, если валидного JSON нет.
    """
    if raw is None:
        raise ValueError("пустой ответ модели")

    text = raw.strip()

    # Снять markdown-ограждение ```json ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Прямой разбор
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: вырезать от первой { или [ до парной закрывающей
    start = _first_bracket(text)
    if start is not None:
        candidate = _balanced_slice(text, start)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"не удалось разобрать JSON из ответа: {raw[:200]!r}")


def _first_bracket(text: str) -> int | None:
    positions = [p for p in (text.find("{"), text.find("[")) if p != -1]
    return min(positions) if positions else None


def _balanced_slice(text: str, start: int) -> str | None:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class LLMClient(abc.ABC):
    """Общий контракт генеративной модели."""

    def __init__(self, model: str, *, max_retries: int = 2, temperature: float = 0.0):
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

    @abc.abstractmethod
    def _raw_complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Один вызов модели, возвращает сырой текст. Реализуется провайдером."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._raw_complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )

    def extract_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict | list:
        """Просит модель вернуть JSON и разбирает его, ретраит при мусоре."""
        sys = system or JSON_SYSTEM
        current = prompt
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = self._raw_complete(current, system=sys, temperature=temperature)
            try:
                return parse_json(raw)
            except ValueError as e:
                last_err = e
                current = (
                    prompt
                    + "\n\nПРЕДЫДУЩИЙ ОТВЕТ БЫЛ НЕВАЛИДНЫМ JSON. "
                    "Верни строго валидный JSON, без текста вокруг."
                )
        raise LLMError(
            f"не удалось получить валидный JSON за {self.max_retries + 1} попыток: {last_err}"
        )
