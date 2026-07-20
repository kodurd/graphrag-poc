"""Чанкинг — чистые функции."""

from __future__ import annotations

import pytest

from graphrag.index.chunk import chunk_text, plan_chunks


def test_short_text_single_chunk():
    assert chunk_text("короткий текст", size=800) == ["короткий текст"]


def test_empty_text_no_chunks():
    assert chunk_text("   ", size=800) == []


def test_long_text_splits_with_overlap():
    text = "".join(str(i % 10) for i in range(2000))  # 2000 символов
    chunks = chunk_text(text, size=800, overlap=120)
    assert len(chunks) >= 2
    # перекрытие: хвост первого чанка == голова второго
    assert chunks[0][-120:] == chunks[1][:120]
    # склейка без overlap восстанавливает исходный текст
    rebuilt = chunks[0] + "".join(c[120:] for c in chunks[1:])
    assert rebuilt == text


def test_overlap_ge_size_raises():
    with pytest.raises(ValueError):
        chunk_text("x" * 100, size=100, overlap=100)


def test_plan_chunks_ids_and_parent():
    nodes = [("task:KAFKA-1", "a" * 1000, "uri1")]
    specs = plan_chunks(nodes, size=400, overlap=50)
    assert specs[0]["id"] == "chunk:task:KAFKA-1#0"
    assert specs[1]["id"] == "chunk:task:KAFKA-1#1"
    assert all(s["parent"] == "task:KAFKA-1" and s["uri"] == "uri1" for s in specs)
