"""Сборка LLM-клиента из конфига — единая точка выбора провайдера."""

from __future__ import annotations

from graphrag.config import LLMConfig
from graphrag.llm.api import APILLMClient
from graphrag.llm.base import LLMClient
from graphrag.llm.ollama import OllamaLLMClient


def build_llm(cfg: LLMConfig, *, role: str = "generation") -> LLMClient:
    """Возвращает клиент под роль ('generation' | 'extraction').

    Провайдер (api/ollama) — из конфига; модель зависит от роли.
    """
    model = cfg.generation_model if role == "generation" else cfg.extraction_model

    if cfg.provider == "api":
        return APILLMClient(
            model,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            max_retries=cfg.max_retries,
            temperature=cfg.temperature,
        )
    if cfg.provider == "ollama":
        return OllamaLLMClient(
            model,
            base=cfg.ollama_base,
            max_retries=cfg.max_retries,
            temperature=cfg.temperature,
        )
    raise ValueError(f"неизвестный LLM-провайдер: {cfg.provider!r} (ожидалось api|ollama)")
