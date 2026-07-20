"""Тонкая обёртка над драйвером Neo4j + помощник для vector index."""

from __future__ import annotations

import re
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase

from graphrag.config import Neo4jConfig

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(value: str) -> str:
    """Пропускает только простые идентификаторы — защита Cypher-интерполяции."""
    if not _IDENT_RE.match(value):
        raise ValueError(f"недопустимый идентификатор для Cypher: {value!r}")
    return value


class Neo4jConnection:
    """Управляет драйвером и даёт удобные методы выполнения Cypher."""

    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.cfg.uri,
                auth=(self.cfg.user, self.cfg.password or ""),
                # Гасим нотификации вида «unknown relationship type» — типы рёбер
                # онтологии появляются в БД по мере загрузки данных, это не ошибка.
                notifications_min_severity="OFF",
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def session(self):
        with self.driver.session(database=self.cfg.database) as sess:
            yield sess

    def run(self, cypher: str, **params) -> list[dict]:
        """Выполняет запрос и возвращает список словарей."""
        with self.session() as sess:
            result = sess.run(cypher, **params)
            return [dict(record) for record in result]

    def verify_connectivity(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def ensure_vector_index(
        self, name: str, label: str, property_key: str, dimensions: int, *, similarity: str = "cosine"
    ) -> None:
        """Идемпотентно создаёт векторный индекс (Neo4j 5.x)."""
        name = _safe_identifier(name)
        label = _safe_identifier(label)
        property_key = _safe_identifier(property_key)
        self.run(
            f"""
            CREATE VECTOR INDEX {name} IF NOT EXISTS
            FOR (n:{label}) ON (n.{property_key})
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: $dims,
                `vector.similarity_function`: $sim
            }} }}
            """,
            dims=dimensions,
            sim=similarity,
        )
