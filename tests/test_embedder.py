"""Эмбеддер и reranker (оффлайн-реализации) — без загрузки моделей."""

from __future__ import annotations

import numpy as np

from graphrag.embeddings.embedder import HashingEmbedder
from graphrag.embeddings.reranker import LexicalReranker


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # векторы уже L2-нормированы


# --- HashingEmbedder ---

def test_encode_fixed_dimension_ru_and_en():
    """Вектор фиксированной размерности и для русского, и для английского."""
    emb = HashingEmbedder(dimension=128)
    vecs = emb.encode(["сервис авторизации упал", "auth service crashed"])
    assert vecs.shape == (2, 128)
    assert vecs.dtype == np.float32


def test_encode_is_deterministic():
    emb = HashingEmbedder(dimension=128)
    v1 = emb.encode(["одинаковый текст"])
    v2 = emb.encode(["одинаковый текст"])
    assert np.allclose(v1, v2)


def test_encode_vectors_are_normalized():
    emb = HashingEmbedder(dimension=64)
    vecs = emb.encode(["непустой текст про kafka"])
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)


def test_overlapping_tokens_more_similar_than_disjoint():
    emb = HashingEmbedder(dimension=512)
    v = emb.encode(
        [
            "kafka connect worker failed",
            "kafka connect task error",   # пересечение по kafka/connect
            "совершенно другая тема суп",  # непересекающийся
        ]
    )
    sim_overlap = _cos(v[0], v[1])
    sim_disjoint = _cos(v[0], v[2])
    assert sim_overlap > sim_disjoint


def test_empty_text_gives_zero_vector():
    """Пустой текст → нулевой вектор (нормализация его не ломает)."""
    emb = HashingEmbedder(dimension=32)
    vecs = emb.encode([""])
    assert vecs.shape == (1, 32)
    assert np.allclose(vecs[0], 0.0)


# --- LexicalReranker ---

def test_lexical_reranker_orders_by_overlap():
    rr = LexicalReranker()
    docs = [
        "непересекающийся документ про погоду",
        "kafka connect упал на воркере",   # ближе к запросу
    ]
    ranked = rr.rerank("kafka connect воркер", docs)
    assert ranked[0][0] == 1  # релевантный документ первым
    assert ranked[0][1] >= ranked[1][1]


def test_lexical_reranker_empty_docs():
    rr = LexicalReranker()
    assert rr.rerank("запрос", []) == []
