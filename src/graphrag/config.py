"""Загрузка конфига: config/settings.yaml + секреты из окружения (.env).

Секреты (ключ API, пароль Neo4j) НИКОГДА не хранятся в YAML — только в env.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class LLMConfig(BaseModel):
    provider: str = "api"  # api | ollama
    generation_model: str = "gpt-4o-mini"
    extraction_model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    ollama_base: str = "http://localhost:11434"
    max_retries: int = 2
    temperature: float = 0.0
    api_key: str | None = None  # из env LLM_API_KEY


class EmbeddingsConfig(BaseModel):
    provider: str = "sentence_transformers"  # sentence_transformers | hashing
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    device: str = "cpu"


class RerankerConfig(BaseModel):
    provider: str = "cross_encoder"  # cross_encoder | lexical
    model: str = "BAAI/bge-reranker-v2-m3"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    database: str = "neo4j"
    password: str | None = None  # из env NEO4J_PASSWORD


class CorpusConfig(BaseModel):
    repo_path: str = "./data/repos/kafka"
    components: list[str] = Field(default_factory=lambda: ["clients", "connect", "streams"])
    since: str = "2024-01-01"


class SourcesConfig(BaseModel):
    jira_base: str = "https://issues.apache.org/jira"
    jira_project: str = "KAFKA"
    confluence_base: str = "https://cwiki.apache.org/confluence"
    confluence_space: str = "KAFKA"
    max_issues: int | None = 500
    max_pages: int | None = 200


class ChunkConfig(BaseModel):
    size: int = 800
    overlap: int = 120


class RetrievalConfig(BaseModel):
    top_k: int = 8
    rerank_top_k: int = 5
    max_hops: int = 3
    # Порог релевантности в шкале активного реранкера (Lexical [0,1] / CrossEncoder
    # логиты). 0 = отключён. Ниже порога вектор/bm25-кандидаты отбрасываются, чтобы
    # система честно воздержалась, а не отвечала по мусору. Граф от порога свободен.
    min_rerank_score: float = 0.0


class EvalConfig(BaseModel):
    # Оценка faithfulness. samples>1 — снижение дисперсии судьи мульти-сэмплом (среднее),
    # temperature — judge-специфичная (отдельно от генерации). Дефолт samples=1 → поведение
    # и стоимость как раньше. Включать только по диагнозу «шум» (см. eval/faith_calib.py).
    faithfulness_judge_samples: int = 1
    faithfulness_judge_temperature: float = 0.3


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Читает YAML и подмешивает секреты из окружения."""
    load_dotenv()  # .env, если есть

    data: dict = {}
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    settings = Settings.model_validate(data)

    # Секреты — только из env, поверх YAML.
    settings.llm.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    # Endpoint/модель провайдера — тоже можно задать через env (напр. DeepSeek),
    # чтобы не править закоммиченный YAML. env имеет приоритет над settings.yaml.
    if os.getenv("LLM_BASE_URL"):
        settings.llm.api_base = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_MODEL"):
        settings.llm.generation_model = os.getenv("LLM_MODEL")
        settings.llm.extraction_model = os.getenv("LLM_MODEL")
    settings.neo4j.password = os.getenv("NEO4J_PASSWORD", "graphrag-local")
    return settings
