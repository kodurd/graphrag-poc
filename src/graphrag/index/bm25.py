"""Полнотекстовый BM25-индекс над чанками (in-memory, rank_bm25)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from graphrag.text import tokenize


class BM25Index:
    """Строит BM25 над списком документов {id, text, uri} и ищет по нему."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self._tokenized = [tokenize(d.get("text", "")) for d in docs]
        # BM25Okapi не любит пустой корпус — держим флаг.
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not self._bm25:
            return []
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        scores = self._bm25.get_scores(q_tokens)
        # Фильтруем по пересечению токенов, а не по знаку score: на малом корпусе
        # BM25-IDF может схлопываться в 0/минус, но ранжирование остаётся валидным.
        candidates = [i for i in range(len(self.docs)) if q_set & set(self._tokenized[i])]
        candidates.sort(key=lambda i: scores[i], reverse=True)
        out = []
        for i in candidates[:top_k]:
            d = self.docs[i]
            out.append({"id": d["id"], "text": d.get("text", ""), "uri": d.get("uri", ""),
                        "score": float(scores[i])})
        return out
