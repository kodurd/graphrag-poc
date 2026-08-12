"""BM25 (чистый) + гибридный retrieval (интеграция)."""

from __future__ import annotations

import pytest

from graphrag.embeddings.embedder import HashingEmbedder
from graphrag.embeddings.reranker import LexicalReranker
from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.index.bm25 import BM25Index
from graphrag.index.vector import VectorIndexer, collect_text_nodes
from graphrag.intermediate import edge, node
from graphrag.retrieval.hybrid import (
    HybridRetriever,
    _neighbor_targets,
    cap_candidates_keep_graph,
    cap_candidates_keep_kip,
    filter_by_threshold,
)
from graphrag.retrieval.router import FACTUAL, MULTIHOP


# --- порог релевантности (чистая функция) ---

def test_threshold_disabled_keeps_all():
    items = [{"id": "a", "source": "vector", "rerank_score": 0.1}]
    assert filter_by_threshold(items, 0.0) == items  # 0 = отключён
    assert filter_by_threshold(items, -1) == items


def test_threshold_drops_low_vector_candidates():
    items = [
        {"id": "hi", "source": "vector", "rerank_score": 0.8},
        {"id": "lo", "source": "bm25", "rerank_score": 0.2},
    ]
    kept = filter_by_threshold(items, 0.5)
    assert [it["id"] for it in kept] == ["hi"]  # низкий отброшен


def test_threshold_all_below_yields_empty():
    items = [{"id": "a", "source": "vector", "rerank_score": 0.1}]
    assert filter_by_threshold(items, 0.5) == []  # -> пусто -> честное «не знаю»


def test_threshold_exempts_graph_candidates():
    items = [
        {"id": "module:connect", "source": "graph", "rerank_score": 0.05},
        {"id": "chunk:lo", "source": "bm25", "rerank_score": 0.05},
    ]
    kept = filter_by_threshold(items, 0.5)
    assert [it["id"] for it in kept] == ["module:connect"]  # граф остаётся


# --- срез top-k с сохранением графа (чистая функция) ---

def _r(id_, source, score):
    return {"id": id_, "source": source, "rerank_score": score}


def test_cap_keeps_all_graph_and_top_k_chunks():
    # 6 чанков ранжированы выше 2 граф-узлов; k=5 → 5 чанков + оба графа выживают
    items = [_r(f"chunk:{i}", "vector", 0.9 - i * 0.1) for i in range(6)]
    items += [_r("module:connect", "graph", 0.05), _r("module:streams", "graph", 0.04)]
    out = cap_candidates_keep_graph(items, 5)
    ids = [c["id"] for c in out]
    assert "module:connect" in ids and "module:streams" in ids  # граф не вытеснен
    assert len([c for c in out if c["source"] != "graph"]) == 5  # чанки кап'нуты до k


def test_cap_upper_bound_is_len_graph_plus_k():
    items = [_r(f"chunk:{i}", "vector", 1.0 - i * 0.1) for i in range(6)]
    items += [_r("module:a", "graph", 0.1), _r("module:b", "graph", 0.1)]
    out = cap_candidates_keep_graph(items, 5)
    assert len(out) == 5 + 2  # len(граф)=2 + k=5


def test_cap_preserves_rerank_order():
    items = [_r("chunk:hi", "vector", 0.9), _r("module:g", "graph", 0.5),
             _r("chunk:lo", "bm25", 0.1)]
    out = cap_candidates_keep_graph(items, 5)
    assert [c["id"] for c in out] == ["chunk:hi", "module:g", "chunk:lo"]  # порядок реранка


def test_cap_without_graph_equals_plain_slice():
    items = [_r(f"chunk:{i}", "vector", 1.0 - i * 0.1) for i in range(8)]
    out = cap_candidates_keep_graph(items, 5)
    assert out == items[:5]  # без графа — обычный срез


# --- срез top-k с резервом под KIP-чанки (чистая функция) ---

def test_kip_reserve_guarantees_page_chunks():
    # 4 task-чанка ранжированы выше 3 page-чанков; k=5, reserve=2 →
    # 2 лучших page доходят, всего 5, порядок реранка внутри групп сохранён
    items = [_r(f"chunk:task:{i}#0", "vector", 0.9 - i * 0.1) for i in range(4)]
    items += [_r(f"chunk:page:P#{i}", "bm25", 0.4 - i * 0.05) for i in range(3)]
    out = cap_candidates_keep_kip(items, 5, 2)
    page_ids = [c["id"] for c in out if c["id"].startswith("chunk:page:")]
    assert len(page_ids) >= 2  # резерв соблюдён
    assert len(out) == 5  # кап не превышен
    assert page_ids == ["chunk:page:P#0", "chunk:page:P#1"]  # лучшие page по рангу


def test_kip_reserve_fills_slots_when_few_page_chunks():
    # только 1 page-чанк, reserve=2 → он сохранён, остаток слотов не теряется
    items = [_r(f"chunk:task:{i}#0", "vector", 0.9 - i * 0.1) for i in range(6)]
    items.insert(3, _r("chunk:page:P#0", "bm25", 0.55))
    out = cap_candidates_keep_kip(items, 5, 2)
    ids = [c["id"] for c in out]
    assert "chunk:page:P#0" in ids  # единственный page сохранён
    assert len(out) == 5  # ни один слот не потерян


def test_kip_reserve_zero_equals_plain_slice():
    items = [_r(f"chunk:page:P#{i}", "vector", 1.0 - i * 0.1) for i in range(8)]
    out = cap_candidates_keep_kip(items, 5, 0)
    assert out == items[:5]  # reserve=0 — прежнее поведение


def test_kip_reserve_no_page_chunks_is_plain_slice():
    # тикет/коммит-вопрос: page-чанков нет → срез неизменен при любом reserve
    items = [_r(f"chunk:task:{i}#0", "vector", 1.0 - i * 0.1) for i in range(8)]
    assert cap_candidates_keep_kip(items, 5, 2) == items[:5]
    assert cap_candidates_keep_kip(items, 5, 5) == items[:5]


def test_kip_reserve_preserves_rank_order_within_groups():
    items = [
        _r("chunk:page:P#0", "bm25", 0.95),
        _r("chunk:task:1#0", "vector", 0.90),
        _r("chunk:page:P#1", "bm25", 0.80),
        _r("chunk:task:2#0", "vector", 0.70),
        _r("chunk:page:P#2", "bm25", 0.60),
    ]
    out = cap_candidates_keep_kip(items, 5, 2)
    page_order = [c["id"] for c in out if c["id"].startswith("chunk:page:")]
    task_order = [c["id"] for c in out if c["id"].startswith("chunk:task:")]
    assert page_order == ["chunk:page:P#0", "chunk:page:P#1", "chunk:page:P#2"]
    assert task_order == ["chunk:task:1#0", "chunk:task:2#0"]  # порядок реранка сохранён


# --- KIP-соседи: цель по seq (чистая функция) ---

def test_neighbor_targets_window1():
    assert _neighbor_targets("page:P", 3, 1) == [("page:P", 2), ("page:P", 4)]


def test_neighbor_targets_window2():
    assert _neighbor_targets("page:P", 5, 2) == [
        ("page:P", 3), ("page:P", 4), ("page:P", 6), ("page:P", 7)
    ]  # +/-2, себя исключая


def test_neighbor_targets_seq0_has_no_negatives():
    assert _neighbor_targets("page:P", 0, 1) == [("page:P", 1)]  # левых соседей нет
    assert _neighbor_targets("page:P", 0, 2) == [("page:P", 1), ("page:P", 2)]


def test_neighbor_targets_window_zero_is_empty():
    assert _neighbor_targets("page:P", 3, 0) == []  # выкл => пусто
    assert _neighbor_targets("page:P", 3, -1) == []


# --- KIP-соседи: расширение с фейковым conn (без Neo4j) ---

class _FakeConn:
    """Мини-стаб: `.run()` пишет вызовы и отдаёт канонические строки соседей."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        return self.rows


def _retriever(conn, kip_neighbors):
    # embedder/reranker не участвуют в _expand_page_neighbors — конструктор их лишь хранит.
    return HybridRetriever(conn, None, None, kip_neighbors=kip_neighbors)


def test_expand_appends_dedups_and_preserves_primary_order():
    cands = [
        {"id": "chunk:page:P#1", "text": "b", "uri": "u", "parent": "page:P",
         "seq": 1, "source": "vector"},
        {"id": "chunk:task:1#0", "text": "t", "uri": "ut", "source": "vector"},
    ]
    # Соседи, что вернул бы Neo4j (включая дубль уже присутствующего P#1 — должен отсеяться).
    rows = [
        {"id": "chunk:page:P#2", "text": "c", "uri": "u", "parent": "page:P", "seq": 2},
        {"id": "chunk:page:P#0", "text": "a", "uri": "u", "parent": "page:P", "seq": 0},
        {"id": "chunk:page:P#1", "text": "b", "uri": "u", "parent": "page:P", "seq": 1},
    ]
    conn = _FakeConn(rows)
    out = _retriever(conn, kip_neighbors=1)._expand_page_neighbors(cands)

    ids = [c["id"] for c in out]
    # primary остаются первыми в исходном порядке
    assert ids[:2] == ["chunk:page:P#1", "chunk:task:1#0"]
    # соседи добавлены в хвост, детерминированно по seq, дубль P#1 отсеян
    assert ids[2:] == ["chunk:page:P#0", "chunk:page:P#2"]
    assert ids.count("chunk:page:P#1") == 1  # дедуп
    assert [c["source"] for c in out[2:]] == ["neighbor", "neighbor"]
    assert len(conn.calls) == 1  # parent/seq в полях => только один запрос (соседи)


def test_expand_disabled_returns_unchanged_and_no_query():
    cands = [{"id": "chunk:page:P#1", "text": "b", "uri": "u",
              "parent": "page:P", "seq": 1, "source": "vector"}]
    conn = _FakeConn([{"id": "chunk:page:P#2", "text": "c", "uri": "u",
                       "parent": "page:P", "seq": 2}])
    out = _retriever(conn, kip_neighbors=0)._expand_page_neighbors(cands)
    assert out == cands  # kip_neighbors=0 => без изменений
    assert conn.calls == []  # и без запроса в БД


def test_expand_ignores_non_page_candidates_without_query():
    cands = [
        {"id": "chunk:task:1#0", "text": "t", "uri": "ut", "source": "vector"},
        {"id": "module:connect", "text": "m", "uri": "graph://x", "source": "graph"},
    ]
    conn = _FakeConn([{"id": "chunk:page:P#2", "text": "c", "uri": "u",
                       "parent": "page:P", "seq": 2}])
    out = _retriever(conn, kip_neighbors=1)._expand_page_neighbors(cands)
    assert out == cands  # нет page-чанков => расширения нет
    assert conn.calls == []  # и запрос соседей не выполнялся


# --- наблюдаемость source ---

def test_source_field_present_and_first_writer_wins():
    # источник наблюдаем; при дубле вектор раньше bm25 -> помечен vector
    items = [
        {"id": "x", "source": "vector", "rerank_score": 0.9},
        {"id": "y", "source": "graph", "rerank_score": 0.4},
    ]
    kept = filter_by_threshold(items, 0.0)
    assert {it["source"] for it in kept} == {"vector", "graph"}


# --- BM25 (чистый) ---

def test_bm25_ranks_overlap_higher():
    idx = BM25Index([
        {"id": "1", "text": "weather dashboard layout colors", "uri": "u1"},
        {"id": "2", "text": "kafka network client reconnect broker", "uri": "u2"},
    ])
    res = idx.search("network client broker", top_k=2)
    assert res and res[0]["id"] == "2"


def test_bm25_empty_corpus():
    assert BM25Index([]).search("что угодно") == []


# --- гибрид (интеграция) ---

def _seed(conn):
    apply_schema(conn)
    load_records(conn, [
        node("Module", "module:clients", {"name": "clients"}),
        node("Module", "module:connect", {"name": "connect"}),
        node("Module", "module:streams", {"name": "streams"}),
        edge("DEPENDS_ON", "module:connect", "module:clients"),
        edge("DEPENDS_ON", "module:streams", "module:clients"),
        node("Task", "task:1",
             {"summary": "NetworkClient reconnect", "description": "clients module broker handling",
              "uri": "https://issues/KAFKA-1"}),
        edge("MENTIONS", "task:1", "module:clients"),
    ])


@pytest.mark.integration
def test_multihop_surfaces_graph_only_modules(neo4j_conn):
    """Multi-hop через граф находит connect/streams, которых нет у вектора."""
    _seed(neo4j_conn)
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker(), max_hops=2)

    result = retr.retrieve("что зависит от clients")
    assert result["route"] == MULTIHOP
    ids = {c["id"] for c in result["candidates"]}
    assert "module:connect" in ids and "module:streams" in ids

    # вектор-only тот же запрос модули не вернёт (у них нет чанков)
    vec_ids = {r["id"] for r in idx.search("что зависит от clients", top_k=8)}
    assert "module:connect" not in vec_ids and "module:streams" not in vec_ids


@pytest.mark.integration
def test_factual_uses_vector_and_bm25(neo4j_conn):
    _seed(neo4j_conn)
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker())

    result = retr.retrieve("что такое NetworkClient")
    assert result["route"] == FACTUAL
    assert result["candidates"], "факт-запрос должен вернуть чанки"
    assert any("networkclient" in c["text"].lower() for c in result["candidates"])


@pytest.mark.integration
def test_candidate_pool_is_pre_rerank(neo4j_conn):
    """_candidate_pool отдаёт пул ДО реранка: без rerank_score, >= числа кандидатов."""
    _seed(neo4j_conn)
    emb = HashingEmbedder(dimension=64)
    idx = VectorIndexer(neo4j_conn, emb)
    idx.ensure_index()
    idx.index_nodes(collect_text_nodes(neo4j_conn))
    neo4j_conn.run("CALL db.awaitIndexes(60)")

    retr = HybridRetriever(neo4j_conn, emb, LexicalReranker())
    q = "что такое NetworkClient"
    route, pool = retr._candidate_pool(q)
    result = retr.retrieve(q)

    assert route == result["route"]
    assert pool and all("rerank_score" not in it for it in pool)  # ещё не ранжирован
    assert len(pool) >= len(result["candidates"])  # top-k — подмножество пула
