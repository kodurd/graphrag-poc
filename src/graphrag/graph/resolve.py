"""Entity resolution (склейка дублей узлов).

Три сигнала: (1) словарь канонических имён (авторитетно, в госухе — реестр
систем), (2) нормализация имён (снимает регистр, стоп-слова, порядок),
(3) embedding-similarity кандидатов (мультиязычный эмбеддер склеивает
«auth-service» и «сервис авторизации»). Слияние — через apoc.refactor.mergeNodes
с сохранением рёбер, поэтому обходы графа не обрываются.
"""

from __future__ import annotations

import numpy as np

from graphrag.embeddings.base import Embedder
from graphrag.graph.connection import Neo4jConnection
from graphrag.text import tokenize

# Стоп-слова, не несущие различающего смысла в именах сервисов.
_STOPWORDS = {
    "сервис", "service", "ас", "система", "system", "module", "модуль",
    "the", "svc", "app", "приложение",
}


def normalize_name(name: str) -> str:
    """Каноническая форма имени: нижний регистр, без стоп-слов, порядок-независимо."""
    tokens = [t for t in tokenize(name) if t not in _STOPWORDS]
    return " ".join(sorted(tokens))


def cluster_key(name: str, canonical: dict[str, str]) -> str:
    """Ключ кластера: словарь канонических имён важнее нормализации."""
    low = name.strip().lower()
    if low in canonical:
        return canonical[low]
    return normalize_name(name) or low


def find_clusters(
    nodes: list[dict],
    *,
    canonical: dict[str, str] | None = None,
    embedder: Embedder | None = None,
    threshold: float = 0.9,
) -> list[list[str]]:
    """Группирует узлы-дубли. nodes: [{id, name}]. Возвращает кластеры id (size>1).

    Сначала по словарю/нормализации; затем оставшиеся синглтоны — по
    embedding-similarity (если дан эмбеддер).
    """
    canonical = {k.lower(): v for k, v in (canonical or {}).items()}
    groups: dict[str, list[str]] = {}
    for n in nodes:
        key = cluster_key(n["name"], canonical)
        groups.setdefault(key, []).append(n["id"])

    # Сортируем id внутри кластера и сами кластеры — survivor (cluster[0]) должен
    # быть детерминированным, а не зависеть от порядка строк Neo4j.
    clusters = [sorted(ids) for ids in groups.values() if len(ids) > 1]

    if embedder is not None:
        singletons = [ids[0] for ids in groups.values() if len(ids) == 1]
        clusters += [sorted(c) for c in _embedding_clusters(nodes, singletons, embedder, threshold)]

    return sorted(clusters)


def _embedding_clusters(
    nodes: list[dict], ids: list[str], embedder: Embedder, threshold: float
) -> list[list[str]]:
    if len(ids) < 2:
        return []
    name_by_id = {n["id"]: n["name"] for n in nodes}
    vecs = embedder.encode([name_by_id[i] for i in ids])
    used: set[int] = set()
    out: list[list[str]] = []
    for i in range(len(ids)):
        if i in used:
            continue
        group = [ids[i]]
        for j in range(i + 1, len(ids)):
            if j in used:
                continue
            if float(np.dot(vecs[i], vecs[j])) >= threshold:
                group.append(ids[j])
                used.add(j)
        if len(group) > 1:
            used.add(i)
            out.append(group)
    return out


class EntityResolver:
    """Находит и сливает дубли узлов Module в графе."""

    def __init__(
        self,
        conn: Neo4jConnection,
        *,
        embedder: Embedder | None = None,
        canonical: dict[str, str] | None = None,
        threshold: float = 0.9,
    ):
        self.conn = conn
        self.embedder = embedder
        self.canonical = canonical or {}
        self.threshold = threshold

    def _modules(self) -> list[dict]:
        return self.conn.run("MATCH (m:Module) RETURN m.id AS id, m.name AS name ORDER BY m.id")

    def resolve(self) -> dict:
        nodes = self._modules()
        clusters = find_clusters(
            nodes, canonical=self.canonical, embedder=self.embedder, threshold=self.threshold
        )
        merged = 0
        for cluster in clusters:
            keep, dups = cluster[0], cluster[1:]
            self._merge(keep, dups)
            merged += len(dups)
        return {"clusters": len(clusters), "merged_nodes": merged}

    def _merge(self, keep_id: str, dup_ids: list[str]) -> None:
        """Сливает dup-узлы в keep через APOC, сохраняя рёбра."""
        if not dup_ids:
            return
        self.conn.run(
            """
            MATCH (keep {id: $keep})
            MATCH (d) WHERE d.id IN $dups
            WITH keep, collect(d) AS ds
            CALL apoc.refactor.mergeNodes([keep] + ds,
                 {properties: 'discard', mergeRels: true, produceSelfRel: false})
            YIELD node
            RETURN node
            """,
            keep=keep_id,
            dups=dup_ids,
        )
