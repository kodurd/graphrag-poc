"""Сборка LLM-клиента из конфига — единая точка выбора провайдера."""

from __future__ import annotations

from graphrag.config import LLMConfig
from graphrag.llm.api import APILLMClient
from graphrag.llm.base import LLMClient
from graphrag.llm.ollama import OllamaLLMClient


def build_llm(cfg: LLMConfig, *, role: str = "generation") -> LLMClient:
    """Возвращает клиент под роль ('generation' | 'extraction' | 'judge').

    generation/extraction — общий провайдер из конфига, модель зависит от роли.
    judge — НЕЗАВИСИМАЯ модель (свой provider/base/key/model из judge_*), чтобы
    оценщик отличался от генератора; при пустых judge_* — fallback на generation.
    Любая другая роль — ошибка (раньше молча уходила в extraction).
    """
    if role == "generation":
        model, provider, api_base, api_key = (
            cfg.generation_model, cfg.provider, cfg.api_base, cfg.api_key)
    elif role == "extraction":
        model, provider, api_base, api_key = (
            cfg.extraction_model, cfg.provider, cfg.api_base, cfg.api_key)
    elif role == "judge":
        # Пустые judge_* => fallback на generation (обратная совместимость).
        model = cfg.judge_model or cfg.generation_model
        provider = cfg.judge_provider or cfg.provider
        api_base = cfg.judge_api_base or cfg.api_base
        api_key = cfg.judge_api_key or cfg.api_key
    else:
        raise ValueError(
            f"неизвестная роль LLM: {role!r} (ожидалось generation|extraction|judge)")

    if provider == "api":
        return APILLMClient(
            model,
            api_base=api_base,
            api_key=api_key,
            max_retries=cfg.max_retries,
            temperature=cfg.temperature,
        )
    if provider == "ollama":
        return OllamaLLMClient(
            model,
            base=cfg.ollama_base,
            max_retries=cfg.max_retries,
            temperature=cfg.temperature,
        )
    raise ValueError(f"неизвестный LLM-провайдер: {provider!r} (ожидалось api|ollama)")
