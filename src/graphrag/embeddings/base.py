"""Абстракции эмбеддера и reranker'а."""

from __future__ import annotations

import abc

import numpy as np


class Embedder(abc.ABC):
    """Кодирует тексты в векторы фиксированной размерности (L2-нормированные)."""

    dimension: int

    @abc.abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Возвращает массив (len(texts), dimension) float32."""


class Reranker(abc.ABC):
    """Переупорядочивает документы по релевантности запросу."""

    @abc.abstractmethod
    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """Возвращает [(исходный_индекс, score)], отсортированный по убыванию score."""
