"""Сервисный слой: переиспользуемое ядро для CLI и UI.

Функции здесь не печатают в stdout и не бросают typer.Exit — они возвращают
данные или бросают ServiceError. CLI (`cli.py`) и Gradio-UI (`ui.py`) —
две тонкие обёртки над одним и тем же кодом.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.config import load_settings


class ServiceError(RuntimeError):
    """Ошибка уровня сервиса (нет ключа, Neo4j недоступен и т.п.)."""


@dataclass
class AskResult:
    text: str
    citations: list[str]
    route: str
    grounded: bool
    hallucinated_citations: list[str] = field(default_factory=list)


def _require_neo4j(conn) -> None:
    # Без префикса команды — его добавляет вызывающий (CLI) единообразно.
    if not conn.verify_connectivity():
        raise ServiceError("Neo4j недоступен — `docker compose up -d`")


def ask_question(question: str) -> AskResult:
    """Гибридный retrieval + генерация ответа с цитированием."""
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.generate.answer import build_context, generate_answer
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        raise ServiceError(
            "не задан LLM_API_KEY (.env). Задайте ключ или provider: ollama."
        )

    with Neo4jConnection(s.neo4j) as conn:
        _require_neo4j(conn)
        retr = HybridRetriever(
            conn,
            build_embedder(s.embeddings),
            build_reranker(s.reranker),
            top_k=s.retrieval.top_k,
            rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops,
            min_rerank_score=s.retrieval.min_rerank_score,
            multihop_full_retrieval=s.retrieval.multihop_full_retrieval,
        )
        retrieved = retr.retrieve(question)
        context = build_context(retrieved["candidates"])
        llm = build_llm(s.llm, role="generation")
        res = generate_answer(llm, question, context)

    return AskResult(
        text=res.text,
        citations=res.citations,
        route=retrieved["route"],
        grounded=res.grounded,
        hallucinated_citations=res.hallucinated_citations,
    )


def analyze_log(text: str, use_llm: bool = False) -> dict:
    """'Лог с ошибкой -> что затронуто' по подграфу зависимостей."""
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.impact import ImpactAnalyzer

    s = load_settings()
    llm = build_llm(s.llm, role="extraction") if use_llm else None
    with Neo4jConnection(s.neo4j) as conn:
        _require_neo4j(conn)
        return ImpactAnalyzer(conn, llm=llm, max_hops=s.retrieval.max_hops).analyze(text)


def check_health() -> bool:
    """Проверить связь с Neo4j."""
    from graphrag.graph import Neo4jConnection

    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        return conn.verify_connectivity()


def info_lines() -> list[str]:
    """Активная конфигурация (без секретов), построчно."""
    s = load_settings()
    return [
        f"LLM:        provider={s.llm.provider} gen={s.llm.generation_model}",
        f"Embeddings: provider={s.embeddings.provider} model={s.embeddings.model}",
        f"Reranker:   provider={s.reranker.provider}",
        f"Neo4j:      {s.neo4j.uri} db={s.neo4j.database}",
        f"Corpus:     {s.corpus.repo_path} components={s.corpus.components}",
    ]
