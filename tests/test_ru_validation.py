"""Русская валидация эмбеддера — на оффлайн-эмбеддере (без ML-моделей)."""

from __future__ import annotations

from graphrag.embeddings.embedder import HashingEmbedder

from eval.ru_validation import compare_embedders, evaluate_recall, load_dataset


def test_load_default_ru_dataset():
    ds = load_dataset()
    assert ds["docs"] and ds["queries"]
    assert all("relevant" in q for q in ds["queries"])


def test_evaluate_recall_finds_relevant_ru_docs():
    """Русский запрос с лексическим пересечением находит нужный документ."""
    ds = load_dataset()
    recall = evaluate_recall(HashingEmbedder(dimension=256), ds, k=3)
    assert recall > 0.0


def test_compare_embedders_returns_score_per_model():
    ds = load_dataset()
    scores = compare_embedders(
        ds,
        {"hash-128": HashingEmbedder(128), "hash-256": HashingEmbedder(256)},
        k=3,
    )
    assert set(scores) == {"hash-128", "hash-256"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_mixed_ru_en_query_does_not_crash():
    ds = {
        "docs": [{"id": "d1", "text": "clients переподключение broker"}],
        "queries": [{"query": "clients reconnect брокер", "relevant": ["d1"]}],
    }
    assert evaluate_recall(HashingEmbedder(128), ds, k=1) >= 0.0
