"""build_llm: ветвление по роли, независимый судья, ошибка на unknown-роли."""

from __future__ import annotations

import pytest

from graphrag.config import LLMConfig, load_settings
from graphrag.llm.factory import build_llm


def _cfg(**kw) -> LLMConfig:
    base = dict(
        provider="api", generation_model="gen-model", extraction_model="ext-model",
        api_base="https://gen/v1", api_key="gen-key",
    )
    base.update(kw)
    return LLMConfig(**base)


def test_generation_and_extraction_roles():
    assert build_llm(_cfg(), role="generation").model == "gen-model"
    assert build_llm(_cfg(), role="extraction").model == "ext-model"


def test_judge_role_uses_independent_judge_fields():
    """Судья бьёт в СВОЙ эндпоинт/ключ/модель, не в generation."""
    c = _cfg(judge_model="qwen", judge_api_base="https://qwen/v1", judge_api_key="qwen-key")
    j = build_llm(c, role="judge")
    assert j.model == "qwen"
    assert j.api_base == "https://qwen/v1"
    assert j.api_key == "qwen-key"


def test_judge_role_falls_back_to_generation_when_unset():
    """Пустые judge_* => судья = generation (обратная совместимость)."""
    j = build_llm(_cfg(), role="judge")
    assert j.model == "gen-model"
    assert j.api_base == "https://gen/v1"
    assert j.api_key == "gen-key"


def test_unknown_role_raises():
    """Неизвестная роль — явная ошибка (раньше молча уходила в extraction)."""
    with pytest.raises(ValueError, match="роль"):
        build_llm(_cfg(), role="bogus")


def test_load_settings_reads_judge_env(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "env-qwen-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://env-qwen/v1")
    monkeypatch.setenv("JUDGE_MODEL", "env-qwen-model")
    s = load_settings()
    assert s.llm.judge_api_key == "env-qwen-key"
    assert s.llm.judge_api_base == "https://env-qwen/v1"
    assert s.llm.judge_model == "env-qwen-model"
