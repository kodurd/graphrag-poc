r"""Confluence-коннектор: страницы/KIP + иерархия и ссылки → JSONL.

Чистый `parse_page` (тестируется на фикстурах) + `ConfluenceConnector`
(пагинация по REST API с бэкоффом).

Производит:
  Page node {id, title, uri}
  Page -LINKS_TO-> Page       (родитель из ancestors)
  Page -MENTIONS-> Task       (ссылки KAFKA-\d+ в теле)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from graphrag.connectors.http import get_json_with_backoff
from graphrag.intermediate import JsonlWriter, edge, node

_ISSUE_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_page(page: dict, *, base_uri: str = "https://cwiki.apache.org/confluence") -> list[dict]:
    """Разбирает одну страницу Confluence в node/edge-записи."""
    records: list[dict] = []
    pid = str(page["id"])
    title = page.get("title", "")
    webui = (page.get("_links") or {}).get("webui", f"/pages/{pid}")
    uri = f"{base_uri}{webui}"
    src = {"source": "confluence", "uri": uri}

    body = ((page.get("body") or {}).get("storage") or {}).get("value", "")
    plain = re.sub(r"\s+", " ", _TAG_RE.sub(" ", body)).strip()
    records.append(
        node(
            "Page",
            f"page:{pid}",
            # text (очищенное тело) нужен для чанкинга вики; ограничим объём узла.
            {"id": pid, "title": title, "uri": uri, "text": plain[:20000]},
            src,
        )
    )

    ancestors = page.get("ancestors") or []
    if ancestors:
        parent_id = str(ancestors[-1]["id"])
        records.append(
            edge("LINKS_TO", f"page:{pid}", f"page:{parent_id}",
                 props={"relation": "child_of"}, source={"source": "confluence"})
        )

    text = _TAG_RE.sub(" ", body)
    for ref in sorted(set(_ISSUE_RE.findall(text))):
        records.append(edge("MENTIONS", f"page:{pid}", f"task:{ref}", source={"source": "confluence"}))

    return records


@dataclass
class ConfluenceConnector:
    """Выгрузка страниц пространства через REST API с пагинацией."""

    base_url: str  # напр. https://cwiki.apache.org/confluence
    space: str  # напр. KAFKA
    page_size: int = 50
    max_pages: int | None = None
    _client: httpx.Client | None = field(default=None, repr=False)

    def _client_or_new(self) -> httpx.Client:
        return self._client or httpx.Client(timeout=60.0)

    def _fetch(self, client: httpx.Client, start: int) -> dict:
        return get_json_with_backoff(
            client,
            f"{self.base_url.rstrip('/')}/rest/api/content",
            params={
                "spaceKey": self.space,
                "start": start,
                "limit": self.page_size,
                "expand": "body.storage,ancestors",
            },
        )

    def extract(self, out_path: str) -> dict:
        client = self._client_or_new()
        stats = {"pages": 0}
        try:
            with JsonlWriter(out_path) as w:
                start = 0
                while True:
                    data = self._fetch(client, start)
                    results = data.get("results", [])
                    if not results:
                        break
                    for page in results:
                        for rec in parse_page(page):
                            w.write(rec)
                            if rec["kind"] == "node" and rec["label"] == "Page":
                                stats["pages"] += 1
                    start += len(results)
                    if self.max_pages and start >= self.max_pages:
                        break
                    if len(results) < self.page_size:
                        break
        finally:
            if self._client is None:
                client.close()
        return stats
