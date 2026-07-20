"""Русская валидация эмбеддера.

Связного русского code+issues+wiki в открытом доступе нет, поэтому русскую
часть пайплайна валидируем на синтетическом срезе: сравниваем recall@k
двух эмбеддеров (bge-m3 vs multilingual-e5). Функции эмбеддер-агностичны
(DI), поэтому тесты идут на оффлайн-эмбеддере без загрузки моделей.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graphrag.embeddings.base import Embedder

from eval.metrics import mean, recall_at_k

DEFAULT_DATASET_PATH = "examples/ru_synthetic.json"


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_recall(embedder: Embedder, dataset: dict, k: int = 3) -> float:
    """Средний recall@k эмбеддера на датасете {docs, queries}."""
    docs = dataset["docs"]
    doc_ids = [d["id"] for d in docs]
    doc_vecs = embedder.encode([d["text"] for d in docs])

    recalls: list[float] = []
    for q in dataset["queries"]:
        qv = embedder.encode([q["query"]])[0]
        sims = doc_vecs @ qv
        ranked = [doc_ids[i] for i in np.argsort(sims)[::-1]]
        recalls.append(recall_at_k(ranked, q["relevant"], k))
    return mean(recalls)


def compare_embedders(
    dataset: dict, embedders: dict[str, Embedder], k: int = 3
) -> dict[str, float]:
    """recall@k по каждому эмбеддеру — для выбора между bge-m3 и e5."""
    return {name: evaluate_recall(emb, dataset, k) for name, emb in embedders.items()}
