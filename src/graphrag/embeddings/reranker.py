"""Reranker: cross-encoder (bge-reranker, ленивый) и лексический (оффлайн)."""

from __future__ import annotations

import numpy as np

from graphrag.config import RerankerConfig
from graphrag.embeddings.base import Reranker
from graphrag.text import tokenize


class LexicalReranker(Reranker):
    """Оффлайн reranker по пересечению токенов (Jaccard). Fallback без ML."""

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        q = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(docs):
            d = set(tokenize(doc))
            union = q | d
            score = len(q & d) / len(union) if union else 0.0
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class CrossEncoderReranker(Reranker):
    """bge-reranker через sentence-transformers CrossEncoder (ленивый, extra `ml`)."""

    def __init__(self, model: str):
        self.model_name = model
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover - зависит от extra `ml`
                raise ImportError(
                    "sentence-transformers не установлен. `uv sync --extra ml` "
                    "или provider: lexical."
                ) from e
            self._model = CrossEncoder(self.model_name, device="cpu")

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        if not docs:
            return []
        self._ensure()
        scores = self._model.predict([(query, d) for d in docs])
        order = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in order]


def build_reranker(cfg: RerankerConfig) -> Reranker:
    if cfg.provider == "lexical":
        return LexicalReranker()
    if cfg.provider == "cross_encoder":
        return CrossEncoderReranker(cfg.model)
    raise ValueError(f"неизвестный reranker-провайдер: {cfg.provider!r}")
