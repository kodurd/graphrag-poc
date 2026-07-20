"""Общая токенизация — единая для индексации (BM25), reranking, сопоставления
имён модулей и entity resolution. Публичная, чтобы модули не зависели от
внутренностей эмбеддера.
"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Разбивает текст на нижне-регистровые алфанумерик-токены."""
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
