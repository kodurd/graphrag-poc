"""Provider-agnostic LLM-слой: API и Ollama взаимозаменяемы."""

from graphrag.llm.base import LLMClient, LLMError, parse_json
from graphrag.llm.factory import build_llm

__all__ = ["LLMClient", "LLMError", "parse_json", "build_llm"]
