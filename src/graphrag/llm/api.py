"""OpenAI-совместимый API-клиент (дев-режим)."""

from __future__ import annotations

import httpx

from graphrag.llm.base import LLMClient, LLMError


class APILLMClient(LLMClient):
    """Клиент к /v1/chat/completions любого OpenAI-совместимого сервера."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str,
        api_key: str | None,
        max_retries: int = 2,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        super().__init__(model, max_retries=max_retries, temperature=temperature)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _raw_complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"ошибка вызова API: {e}") from e

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"неожиданный формат ответа API: {data}") from e
