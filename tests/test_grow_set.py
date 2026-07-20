"""U3: генератор роста набора — чистая merge_dedup над нормализацией вопросов."""

from __future__ import annotations

from eval.grow_set import merge_dedup


def test_merge_dedup_drops_duplicate_by_normalize():
    existing = [{"question": "Почему падает?", "source_id": "task:1"}]
    generated = [
        {"question": "почему падает?  ", "source_id": "task:9"},  # дубль по нормализации
        {"question": "Что сломает изменение?", "source_id": "task:2"},
    ]
    out = merge_dedup(existing, generated)
    assert [x["question"] for x in out] == ["Почему падает?", "Что сломает изменение?"]


def test_merge_dedup_keeps_existing_first_and_adds_new():
    existing = [{"question": "A", "source_id": "t1"}]
    generated = [{"question": "B", "source_id": "t2"}]
    out = merge_dedup(existing, generated)
    assert len(out) == 2 and out[0]["question"] == "A" and out[1]["question"] == "B"


def test_merge_dedup_case_and_space_insensitive():
    existing = [{"question": "Hello World", "source_id": "t1"}]
    generated = [{"question": "  hello world ", "source_id": "t2"}]
    assert len(merge_dedup(existing, generated)) == 1  # регистр/пробелы не создают дублей


def test_merge_dedup_empty_generated_unchanged():
    existing = [{"question": "A", "source_id": "t1"}]
    assert merge_dedup(existing, []) == existing


def test_merge_dedup_dedups_within_generated():
    out = merge_dedup([], [{"question": "X", "source_id": "a"},
                           {"question": "x ", "source_id": "b"}])
    assert len(out) == 1
