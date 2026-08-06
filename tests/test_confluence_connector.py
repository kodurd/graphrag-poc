"""Confluence-коннектор — парсер страниц на фикстурах + построение запроса _fetch."""

from __future__ import annotations

import httpx

from graphrag.connectors.confluence import ConfluenceConnector, parse_page

_BASE = "https://cwiki.apache.org/confluence"


def _by(records, kind, **match):
    return [r for r in records if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())]


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _page(n: int) -> dict:
    return {"id": str(1000 + n), "title": f"KIP-{n}", "body": {"storage": {"value": "x"}}}


SAMPLE_PAGE = {
    "id": "12345",
    "title": "KIP-500: Replace ZooKeeper",
    "body": {"storage": {"value": "<p>See KAFKA-101 and KAFKA-205 for details.</p>"}},
    "ancestors": [{"id": "10000"}, {"id": "11111"}],
    "_links": {"webui": "/display/KAFKA/KIP-500"},
}


def test_parse_page_node_and_uri():
    recs = parse_page(SAMPLE_PAGE)
    pages = _by(recs, "node", label="Page")
    assert len(pages) == 1
    p = pages[0]
    assert p["id"] == "page:12345"
    assert p["props"]["title"].startswith("KIP-500")
    assert p["props"]["uri"].endswith("/display/KAFKA/KIP-500")


def test_parse_page_parent_hierarchy():
    """Родитель — последний ancestor (ближайший)."""
    recs = parse_page(SAMPLE_PAGE)
    links = _by(recs, "edge", type="LINKS_TO")
    assert any(e["to"] == "page:11111" for e in links)


def test_parse_page_mentions_issues():
    recs = parse_page(SAMPLE_PAGE)
    mentions = _by(recs, "edge", type="MENTIONS")
    targets = {e["to"] for e in mentions}
    assert "task:KAFKA-101" in targets
    assert "task:KAFKA-205" in targets


def test_parse_page_without_ancestors():
    """Страница без родителей → нет LINKS_TO, не падение."""
    page = {"id": "1", "title": "root", "body": {"storage": {"value": "no refs"}}}
    recs = parse_page(page)
    assert _by(recs, "edge", type="LINKS_TO") == []
    assert _by(recs, "node", label="Page")


# --- _fetch: построение запроса (сеть замокана httpx.MockTransport) ---


def test_fetch_with_cql_hits_search_endpoint(tmp_path):
    """CQL задан → запрос на /search с параметром cql, без spaceKey."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    conn = ConfluenceConnector(
        _BASE, "KAFKA", cql="type=page AND ancestor=12345", _client=_client(handler)
    )
    conn.extract(str(tmp_path / "out.jsonl"))

    assert len(seen) == 1
    req = seen[0]
    assert req.url.path.endswith("/rest/api/content/search")
    assert req.url.params["cql"] == "type=page AND ancestor=12345"
    assert "spaceKey" not in req.url.params


def test_fetch_without_cql_uses_spacekey(tmp_path):
    """CQL не задан → прежний путь /rest/api/content по spaceKey (backward-compat)."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    conn = ConfluenceConnector(_BASE, "KAFKA", _client=_client(handler))
    conn.extract(str(tmp_path / "out.jsonl"))

    assert len(seen) == 1
    req = seen[0]
    assert req.url.path.endswith("/rest/api/content")
    assert not req.url.path.endswith("/search")
    assert req.url.params["spaceKey"] == "KAFKA"
    assert "cql" not in req.url.params


def test_fetch_pagination_concatenates_and_caps_max_pages(tmp_path):
    """Несколько страниц ответа склеиваются; max_pages ограничивает выборку."""
    starts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        starts.append(start)
        results = [_page(start + i) for i in range(2)]
        return httpx.Response(200, json={"results": results})

    conn = ConfluenceConnector(
        _BASE, "KAFKA", page_size=2, max_pages=4, _client=_client(handler)
    )
    stats = conn.extract(str(tmp_path / "out.jsonl"))

    # 2 страницы по 2 результата = 4 Page-узла, затем стоп по max_pages.
    assert stats["pages"] == 4
    assert starts == [0, 2]


def test_fetch_retries_on_429(tmp_path):
    """429 → бэкофф-ретрай через get_json_with_backoff (sleep инъектирован)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"results": []})

    conn = ConfluenceConnector(
        _BASE, "KAFKA", _client=_client(handler), _sleep=lambda _: None
    )
    conn.extract(str(tmp_path / "out.jsonl"))

    assert calls["n"] == 2
