"""human_gold: отбор (refusal + макс-Δ) и валидация человеческой разметки."""

from __future__ import annotations

import json

import pytest

from eval.human_gold import (
    is_refusal,
    load_gold,
    select_gold,
    select_nonrefusal_gold,
)


def _r(sid, answer, abstained=False, route="mixed"):
    return {"source_id": sid, "question": f"q-{sid}", "answer": answer, "route": route,
            "abstained": {"faithfulness": abstained}, "context_texts": ["ctx"]}


def test_is_refusal_by_flag_and_text():
    assert is_refusal(_r("t1", "любой", abstained=True))
    assert is_refusal(_r("t2", "На основе контекста невозможно оценить это."))
    assert not is_refusal(_r("t3", "Чтобы починить, сделай X [источник: u]."))


def test_select_prioritizes_refusals():
    recs = [
        _r("t1", "невозможно оценить"),          # refusal (text)
        _r("t2", "нормальный ответ про X"),      # not
        _r("t3", "любой", abstained=True),       # refusal (flag)
    ]
    sel = select_gold(recs, cross_results=None, limit=2)
    sids = {r["source_id"] for r in sel}
    assert sids == {"t1", "t3"}  # оба refusal, non-refusal t2 не влез в лимит 2


def test_select_fills_with_highest_judge_disagreement():
    recs = [_r("t1", "нормальный A"), _r("t2", "нормальный B"), _r("t3", "нормальный C")]
    # t2 — судьи расходятся сильнее всего
    cross = [
        {"source_id": "t1", "question": "q-t1", "deepseek": {"faithfulness": 1.0},
         "qwen": {"faithfulness": 0.9}},
        {"source_id": "t2", "question": "q-t2", "deepseek": {"faithfulness": 1.0},
         "qwen": {"faithfulness": 0.1}},  # Δ=0.9
        {"source_id": "t3", "question": "q-t3", "deepseek": {"faithfulness": 1.0},
         "qwen": {"faithfulness": 0.8}},
    ]
    sel = select_gold(recs, cross_results=cross, limit=1)
    assert sel[0]["source_id"] == "t2"


def test_select_nonrefusal_excludes_refusals():
    recs = [
        _r("t1", "невозможно оценить"),        # refusal -> исключён
        _r("t2", "нормальный ответ A"),
        _r("t3", "любой", abstained=True),     # refusal (flag) -> исключён
        _r("t4", "нормальный ответ B"),
    ]
    sel = select_nonrefusal_gold(recs, cross_results=None, limit=10)
    assert {r["source_id"] for r in sel} == {"t2", "t4"}


def test_select_nonrefusal_spreads_across_routes():
    recs = [_r(f"m{i}", "норм", route="mixed") for i in range(5)] + \
           [_r("f1", "норм", route="factual"), _r("h1", "норм", route="multihop")]
    sel = select_nonrefusal_gold(recs, cross_results=None, limit=3, per_route=True)
    # round-robin по маршрутам: должны попасть разные маршруты, не только mixed
    routes = {r["route"] for r in sel}
    assert len(routes) >= 2 and len(sel) == 3


def test_load_gold_valid(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({
        "task:KAFKA-1": {"faithfulness": 1.0, "answer_relevance": 0.5, "context_precision": 0.3}
    }), encoding="utf-8")
    assert load_gold(str(p))["task:KAFKA-1"]["faithfulness"] == 1.0


def test_load_gold_rejects_out_of_range(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({
        "task:KAFKA-1": {"faithfulness": 1.5, "answer_relevance": 0.5, "context_precision": 0.3}
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="0..1"):
        load_gold(str(p))
