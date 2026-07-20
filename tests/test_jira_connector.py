"""JIRA-коннектор — парсер тикетов и бэкофф на 429."""

from __future__ import annotations

import httpx
import pytest

from graphrag.connectors.http import get_json_with_backoff
from graphrag.connectors.jira import parse_issue


def _records_by_kind(records, kind, **match):
    out = []
    for r in records:
        if r["kind"] != kind:
            continue
        if all(r.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


SAMPLE_ISSUE = {
    "key": "KAFKA-101",
    "fields": {
        "summary": "Producer fails on retry",
        "description": "NPE в auth-service при переподключении",
        "status": {"name": "Resolved"},
        "assignee": {"displayName": "Ann Dev", "name": "ann"},
        "components": [{"name": "clients"}],
        "issuelinks": [
            {"type": {"name": "Duplicate"}, "outwardIssue": {"key": "KAFKA-50"}},
            {"type": {"name": "Problem/Incident", "outward": "causes"},
             "inwardIssue": {"key": "KAFKA-70"}},
        ],
        "updated": "2024-03-05T12:00:00.000+0000",
    },
}


def test_parse_issue_task_node():
    recs = parse_issue(SAMPLE_ISSUE)
    tasks = _records_by_kind(recs, "node", label="Task")
    assert len(tasks) == 1
    t = tasks[0]
    assert t["id"] == "task:KAFKA-101"
    assert t["props"]["status"] == "Resolved"
    assert t["source"]["uri"].endswith("/KAFKA-101")


def test_parse_issue_assignee_and_component():
    recs = parse_issue(SAMPLE_ISSUE)
    persons = _records_by_kind(recs, "node", label="Person")
    assert persons and persons[0]["props"]["name"] == "Ann Dev"
    assigned = _records_by_kind(recs, "edge", type="ASSIGNED_TO")
    assert assigned and assigned[0]["to"].startswith("person:")
    mentions = _records_by_kind(recs, "edge", type="MENTIONS")
    assert any(e["to"] == "module:clients" for e in mentions)


def test_parse_issue_links():
    recs = parse_issue(SAMPLE_ISSUE)
    dup = _records_by_kind(recs, "edge", type="DUPLICATES")
    assert any(e["to"] == "task:KAFKA-50" for e in dup)
    links = _records_by_kind(recs, "edge", type="LINKS_TO")
    assert any(e["to"] == "task:KAFKA-70" for e in links)


def test_parse_issue_without_assignee():
    """Тикет без assignee → нет ASSIGNED_TO, не падение."""
    issue = {"key": "KAFKA-9", "fields": {"summary": "x", "status": {"name": "Open"}}}
    recs = parse_issue(issue)
    assert _records_by_kind(recs, "edge", type="ASSIGNED_TO") == []
    assert _records_by_kind(recs, "node", label="Task")


def test_backoff_retries_on_429_then_succeeds():
    """429 → бэкофф и повтор; на 2-й попытке 200 (sleep замокан)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = get_json_with_backoff(client, "http://x/api", sleep=lambda _: None)
    assert result == {"ok": True}
    assert calls["n"] == 2


def test_backoff_retries_on_transport_error():
    """ConnectError/ReadTimeout тоже ретраятся (то, ради чего бэкофф и нужен)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = get_json_with_backoff(client, "http://x/api", sleep=lambda _: None)
    assert result == {"ok": True}
    assert calls["n"] == 2


def test_backoff_exhaustion_raises():
    """После исчерпания попыток бросается ошибка, а не тихий None."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        get_json_with_backoff(client, "http://x/api", max_retries=1, sleep=lambda _: None)
