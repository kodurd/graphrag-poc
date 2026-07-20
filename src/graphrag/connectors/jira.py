"""JIRA-коннектор: тикеты + связи → промежуточный JSONL.

Чистый `parse_issue` (тестируется на фикстурах) + `JiraConnector` (пагинация
по REST API с бэкоффом). Формат — Apache JIRA REST v2.

Производит:
  Task node {key, summary, status, description}
  Person node (assignee) + Task -ASSIGNED_TO-> Person
  Task -DUPLICATES-> Task            (issuelink типа Duplicate)
  Task -LINKS_TO-> Task              (прочие issuelinks, prop link_type)
  Task -MENTIONS-> Module            (по components)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from graphrag.connectors.http import get_json_with_backoff
from graphrag.intermediate import JsonlWriter, edge, node


def parse_issue(issue: dict, *, base_uri: str = "https://issues.apache.org/jira/browse") -> list[dict]:
    """Разбирает один тикет JIRA в список node/edge-записей."""
    records: list[dict] = []
    key = issue["key"]
    fields = issue.get("fields", {})
    uri = f"{base_uri}/{key}"
    src = {"source": "jira", "uri": uri, "date": fields.get("updated") or fields.get("created")}

    records.append(
        node(
            "Task",
            f"task:{key}",
            {
                "key": key,
                "summary": fields.get("summary", ""),
                "status": (fields.get("status") or {}).get("name", ""),
                "description": fields.get("description") or "",
            },
            src,
        )
    )

    assignee = fields.get("assignee")
    if assignee and assignee.get("displayName"):
        name = assignee["displayName"]
        pid = f"person:{assignee.get('name') or name}"
        records.append(node("Person", pid, {"name": name}, {"source": "jira"}))
        records.append(edge("ASSIGNED_TO", f"task:{key}", pid, source=src))

    for comp in fields.get("components", []) or []:
        cname = comp.get("name")
        if cname:
            records.append(edge("MENTIONS", f"task:{key}", f"module:{cname}", source={"source": "jira"}))

    for link in fields.get("issuelinks", []) or []:
        ltype = (link.get("type") or {}).get("name", "")
        other = link.get("outwardIssue") or link.get("inwardIssue")
        if not other:
            continue
        other_key = other.get("key")
        if not other_key:
            continue
        rel = "DUPLICATES" if ltype.lower() == "duplicate" else "LINKS_TO"
        props = {} if rel == "DUPLICATES" else {"link_type": ltype}
        records.append(edge(rel, f"task:{key}", f"task:{other_key}", props=props, source={"source": "jira"}))

    return records


@dataclass
class JiraConnector:
    """Выгрузка тикетов проекта через REST API с пагинацией."""

    base_url: str  # напр. https://issues.apache.org/jira
    project: str  # напр. KAFKA
    page_size: int = 100
    max_issues: int | None = None
    extra_jql: str = ""
    _client: httpx.Client | None = field(default=None, repr=False)

    def _client_or_new(self) -> httpx.Client:
        return self._client or httpx.Client(timeout=60.0)

    def _search(self, client: httpx.Client, start_at: int) -> dict:
        jql = f"project = {self.project}"
        if self.extra_jql:
            jql += f" AND {self.extra_jql}"
        return get_json_with_backoff(
            client,
            f"{self.base_url.rstrip('/')}/rest/api/2/search",
            params={
                "jql": jql,
                "startAt": start_at,
                "maxResults": self.page_size,
                "fields": "summary,description,assignee,status,components,issuelinks,updated,created",
            },
        )

    def extract(self, out_path: str) -> dict:
        client = self._client_or_new()
        stats = {"tasks": 0}
        try:
            with JsonlWriter(out_path) as w:
                start = 0
                while True:
                    data = self._search(client, start)
                    issues = data.get("issues", [])
                    if not issues:
                        break
                    for issue in issues:
                        for rec in parse_issue(issue):
                            w.write(rec)
                            if rec["kind"] == "node" and rec["label"] == "Task":
                                stats["tasks"] += 1
                    start += len(issues)
                    if self.max_issues and start >= self.max_issues:
                        break
                    if start >= data.get("total", start):
                        break
        finally:
            if self._client is None:
                client.close()
        return stats
