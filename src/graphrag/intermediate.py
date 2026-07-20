"""Промежуточное хранилище — контракт между Слоем 1 (коннекторы) и Слоем 2 (граф).

Коннекторы пишут нормализованный JSONL из node/edge-записей с метаданными.
Построение графа читает ТОЛЬКО отсюда. Это же — основа инкрементальности (диф по id).

Формат записи:
  node: {"kind":"node","label":..,"id":..,"props":{..},"source":{..}}
  edge: {"kind":"edge","type":..,"from":..,"to":..,"props":{..},"source":{..}}

`source` несёт метаданные происхождения: {source, date, author, uri} — для
цитирования (uri) и инкрементальных апдейтов (date).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def node(label: str, id: str, props: dict | None = None, source: dict | None = None) -> dict:
    return {"kind": "node", "label": label, "id": id, "props": props or {}, "source": source or {}}


def edge(
    type: str,
    from_id: str,
    to_id: str,
    props: dict | None = None,
    source: dict | None = None,
) -> dict:
    return {
        "kind": "edge",
        "type": type,
        "from": from_id,
        "to": to_id,
        "props": props or {},
        "source": source or {},
    }


class JsonlWriter:
    """Пишет записи в JSONL (по одной на строку). Контекст-менеджер."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()

    def write(self, record: dict) -> None:
        assert self._fh is not None, "writer не открыт (используйте with)"
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
