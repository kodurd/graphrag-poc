"""Локальный Ollama-клиент (swap-in для честной «локальной LLM» на демо)."""

from __future__ import annotations

import httpx

from graphrag.llm.base import LLMClient, LLMError


class OllamaLLMClient(LLMClient):
    """Клиент к локальному Ollama (/api/chat)."""

    def __init__(
        self,
        model: str,
        *,
        base: str = "http://localhost:11434",
        max_retries: int = 2,
        temperature: float = 0.0,
        timeout: float = 300.0,
    ):
        super().__init__(model, max_retries=max_retries, temperature=temperature)
        self.base = base.rstrip("/")
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

        options: dict = {"temperature": self.temperature if temperature is None else temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            resp = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"ошибка вызова Ollama: {e}") from e

        data = resp.json()
        try:
            return data["message"]["content"]
        except KeyError as e:
            raise LLMError(f"неожиданный формат ответа Ollama: {data}") from e
