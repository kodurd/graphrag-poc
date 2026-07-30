"""Чистое ядро диагностики покрытия: отбор, парс, классификация, агрегат."""

from __future__ import annotations

from eval.coverage_diag import aggregate, classify, parse_coverage, select_failed_howto


def _rec(route, faith=None, abstained=False):
    return {"route": route, "metrics": {"faithfulness": faith},
            "abstained": {"faithfulness": abstained}}


# --- select_failed_howto (None-guard критичен) ---

def test_select_takes_low_faith_and_abstained_howto():
    recs = [
        _rec("mixed", faith=0.1),                 # взят (низкая faith)
        _rec("multihop", faith=None, abstained=True),  # взят (воздержание, faith null — None-guard)
        _rec("factual", faith=0.1),               # пропущен (не how-to-маршрут)
        _rec("mixed", faith=0.9),                 # пропущен (высокая faith)
    ]
    sel = select_failed_howto(recs)
    assert len(sel) == 2
    routes = {r["route"] for r in sel}
    assert routes == {"mixed", "multihop"}


def test_select_none_faith_without_abstention_not_crashing():
    # faith null, но не воздержание → None-guard не роняет, запись не взята
    assert select_failed_howto([_rec("mixed", faith=None, abstained=False)]) == []


# --- parse_coverage (None ≠ False) ---

def test_parse_coverage_yes_no_and_fence():
    assert parse_coverage('{"sufficient": true}') == {"sufficient": True}
    assert parse_coverage('```json\n{"sufficient": false}\n```') == {"sufficient": False}


def test_parse_coverage_none_on_garbage_or_missing():
    assert parse_coverage("совсем не json") is None
    assert parse_coverage('{"что-то": 1}') is None  # нет ключа sufficient
    assert parse_coverage('{"sufficient": "yes"}') is None  # не bool


# --- classify (judge_failed при None) ---

def test_classify_four_classes():
    assert classify(True, False) == "ranked"          # хватает прод
    assert classify(False, True) == "unretrieved"     # хватает широкого, не прод
    assert classify(False, False) == "not_surfaced"   # не поднят даже широко
    assert classify(None, True) == "judge_failed"     # сбой судьи на прод-уровне
    assert classify(False, None) == "judge_failed"    # сбой на широком


# --- aggregate (не-ranked база, зона неопределённости) ---

def test_aggregate_verdict_sources():
    classes = ["not_surfaced"] * 7 + ["unretrieved"] * 3  # 0.7 not_surfaced
    v = aggregate(classes)
    assert v["verdict"] == "sources" and v["base"] == 10


def test_aggregate_verdict_retrieval():
    classes = ["not_surfaced"] * 2 + ["unretrieved"] * 8  # 0.2
    assert aggregate(classes)["verdict"] == "retrieval"


def test_aggregate_verdict_ambiguous_band():
    classes = ["not_surfaced"] * 5 + ["unretrieved"] * 5  # 0.5 → зона неопределённости
    assert aggregate(classes)["verdict"] == "ambiguous"


def test_aggregate_judge_failed_out_of_denominator():
    classes = ["not_surfaced"] * 7 + ["unretrieved"] * 3 + ["judge_failed"] * 5
    v = aggregate(classes)
    assert v["base"] == 10 and v["judge_failed"] == 5  # judge_failed вне знаменателя
    assert v["verdict"] == "sources"


def test_aggregate_ranked_excluded_from_base():
    classes = ["ranked"] * 4 + ["not_surfaced"] * 6 + ["unretrieved"] * 4
    v = aggregate(classes)
    assert v["base"] == 10  # ranked вне базы
    assert v["not_surfaced_frac"] == 0.6


def test_aggregate_empty_base_is_ambiguous():
    v = aggregate(["ranked", "judge_failed"])
    assert v["base"] == 0 and v["verdict"] == "ambiguous"
