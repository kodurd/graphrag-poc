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


# --- Секционный чанкинг Page/KIP (U2) --------------------------------------


def test_page_sections_one_chunk_each():
    """Happy: две коротких секции по маркерам `## ` → два чанка, по одному на секцию."""
    text = "## A\nsecA (short)\n## B\nsecB (short)"
    specs = plan_chunks([("page:42", text, "uri42")], size=800, overlap=120)
    assert len(specs) == 2
    # каждый чанк — цельная секция (границы по разделам, тела не перемешаны)
    assert specs[0]["text"] == "## A\nsecA (short)"
    assert specs[1]["text"] == "## B\nsecB (short)"
    assert [s["seq"] for s in specs] == [0, 1]
    assert specs[0]["id"] == "chunk:page:42#0"
    assert specs[1]["id"] == "chunk:page:42#1"
    assert all(s["parent"] == "page:42" and s["uri"] == "uri42" for s in specs)


def test_page_long_section_windowed_within_section():
    """Длинная секция (> size) режется окном ВНУТРИ себя, не смешивая соседнюю секцию."""
    long_body = "x" * 1000  # заведомо больше size=400
    text = f"## Big\n{long_body}\n## Small\ntiny tail"
    specs = plan_chunks([("page:7", text, None)], size=400, overlap=50)
    # длинная секция даёт несколько чанков + короткая одна → минимум 3
    assert len(specs) >= 3
    # последний чанк — целиком короткая секция, без примеси длинной
    assert specs[-1]["text"] == "## Small\ntiny tail"
    assert "x" not in specs[-1]["text"]
    # ни один чанк длинной секции не содержит текста короткой
    big_chunks = specs[:-1]
    assert all("tiny tail" not in c["text"] for c in big_chunks)
    # seq сквозной по всем секциям
    assert [s["seq"] for s in specs] == list(range(len(specs)))


def test_non_page_node_uses_blind_window_unchanged():
    """Не-Page узел (task) → прежний слепой оконный путь, байт-идентично старому поведению."""
    nodes = [("task:KAFKA-1", "a" * 1000, "uri1")]
    specs = plan_chunks(nodes, size=400, overlap=50)
    # эталон: слепое окно напрямую через chunk_text
    expected = chunk_text("a" * 1000, size=400, overlap=50)
    assert [s["text"] for s in specs] == expected
    assert [s["id"] for s in specs] == [
        f"chunk:task:KAFKA-1#{i}" for i in range(len(expected))
    ]


def test_page_without_markers_falls_back_to_blind_window():
    """Backward-compat: Page без `## ` маркеров → слепой оконный путь, без падения."""
    flat = "a" * 1000  # текст страницы, загруженной до U1 — без секций
    specs = plan_chunks([("page:99", flat, "uri99")], size=400, overlap=50)
    expected = chunk_text(flat, size=400, overlap=50)
    assert [s["text"] for s in specs] == expected
    assert [s["id"] for s in specs] == [
        f"chunk:page:99#{i}" for i in range(len(expected))
    ]
