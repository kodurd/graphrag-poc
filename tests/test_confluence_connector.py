"""Confluence-коннектор — парсер страниц на фикстурах."""

from __future__ import annotations

from graphrag.connectors.confluence import parse_page


def _by(records, kind, **match):
    return [r for r in records if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())]


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
