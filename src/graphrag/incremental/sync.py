"""Инкрементальный ре-sync (без community summaries).

Диф промежуточного JSONL по id + контент-хэшу → пересчитываются только
добавленные/изменённые записи, удалённые снимаются вместе с висячими рёбрами
и чанками. Пересчёта всего графа нет — это и есть аргумент про «дьявол в
обновлениях».
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from graphrag.graph.connection import Neo4jConnection
from graphrag.graph.schema import validate_rel
from graphrag.graph.skeleton import load_records


def record_key(r: dict) -> str:
    """Стабильный ключ записи: node:<id> либо edge:<from>|<type>|<to>."""
    if r["kind"] == "node":
        return f"node:{r['id']}"
    return f"edge:{r['from']}|{r['type']}|{r['to']}"


def content_hash(r: dict) -> str:
    """Хэш содержимого записи (меняется только при смене смысловых полей)."""
    if r["kind"] == "node":
        payload = {"label": r["label"], "props": r.get("props", {})}
    else:
        payload = {"type": r["type"], "props": r.get("props", {})}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def build_manifest(records: Iterable[dict]) -> dict[str, str]:
    """Манифест {ключ записи -> контент-хэш} для дифа."""
    return {record_key(r): content_hash(r) for r in records}


def diff_manifests(old: dict[str, str], new: dict[str, str]) -> dict[str, set[str]]:
    """Разница манифестов: added / changed / removed / unchanged (множества ключей)."""
    old_keys, new_keys = set(old), set(new)
    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys
    changed = {k for k in common if old[k] != new[k]}
    unchanged = common - changed
    return {"added": added, "changed": changed, "removed": removed, "unchanged": unchanged}


def load_manifest(path: str | Path) -> dict[str, str]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_manifest(manifest: dict[str, str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


class IncrementalSync:
    """Применяет диф записей к графу, трогая только изменённое."""

    def __init__(self, conn: Neo4jConnection):
        self.conn = conn

    def apply(self, new_records: list[dict], prev_manifest: dict[str, str]) -> tuple[dict, dict]:
        """Возвращает (stats, new_manifest). Трогает только added/changed/removed."""
        new_manifest = build_manifest(new_records)
        diff = diff_manifests(prev_manifest, new_manifest)
        by_key = {record_key(r): r for r in new_records}

        to_upsert = [by_key[k] for k in (diff["added"] | diff["changed"])]
        if to_upsert:
            load_records(self.conn, to_upsert)

        for key in diff["removed"]:
            self._remove(key)

        stats = {
            "added": len(diff["added"]),
            "changed": len(diff["changed"]),
            "removed": len(diff["removed"]),
            "unchanged": len(diff["unchanged"]),
            "touched": len(diff["added"]) + len(diff["changed"]) + len(diff["removed"]),
        }
        return stats, new_manifest

    def _remove(self, key: str) -> None:
        if key.startswith("node:"):
            node_id = key[len("node:") :]
            # снять узел вместе с его чанками — висячих рёбер не остаётся
            self.conn.run(
                """
                MATCH (n {id: $id})
                OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(n)
                DETACH DELETE n, c
                """,
                id=node_id,
            )
        else:
            _, rest = key.split(":", 1)
            frm, rel, to = rest.split("|", 2)
            validate_rel(rel)  # защита от инъекции типа
            self.conn.run(
                f"MATCH ({{id: $f}})-[r:{rel}]->({{id: $t}}) DELETE r",
                f=frm,
                t=to,
            )
