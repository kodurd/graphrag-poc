"""Реализации эмбеддера: bge-m3 (реальный, ленивый) и Hashing (оффлайн).

HashingEmbedder детерминирован, без внешних зависимостей — им работают тесты
и оффлайн-режим на слабом железе. SentenceTransformerEmbedder грузит bge-m3
лениво (extra `ml`).
"""

from __future__ import annotations

import hashlib

import numpy as np

from graphrag.config import EmbeddingsConfig
from graphrag.embeddings.base import Embedder
from graphrag.text import tokenize


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder(Embedder):
    """Детерминированный хеш-эмбеддер (bag-of-tokens в фиксированную размерность).

    Не «умный», но стабильный и мгновенный — годится для тестов и как оффлайн
    fallback, когда bge-m3 недоступен. Одинаковый текст → одинаковый вектор;
    пересекающиеся токены → ненулевая косинусная близость.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                out[i, idx] += 1.0
        return _l2_normalize(out)


class SentenceTransformerEmbedder(Embedder):
    """bge-m3 (или иной sentence-transformers) на CPU. Модель грузится лениво."""

    def __init__(self, model: str, *, device: str = "cpu", dimension: int = 1024):
        self.model_name = model
        self.device = device
        self.dimension = dimension
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - зависит от extra `ml`
                raise ImportError(
                    "sentence-transformers не установлен. Поставьте extra: "
                    "`uv sync --extra ml`, либо используйте provider: hashing."
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.astype(np.float32)


def build_embedder(cfg: EmbeddingsConfig) -> Embedder:
    if cfg.provider == "hashing":
        return HashingEmbedder(dimension=cfg.dimension)
    if cfg.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            cfg.model, device=cfg.device, dimension=cfg.dimension
        )
    raise ValueError(f"неизвестный embeddings-провайдер: {cfg.provider!r}")
