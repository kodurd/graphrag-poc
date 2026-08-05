"""kip_delta: первичный лифт vs диагностика + сторожа R7/R8/R9. Синтетика, без I/O."""

from __future__ import annotations

from eval.kip_delta import (
    attach_qwen_faith,
    classify_transition,
    diagnostic_subset,
    displacement_check,
    kip_in_topk,
    noise_floor_ok,
    primary_lift,
    reachability_gate,
    report,
    unmatched_unanswered,
)


def _rec(sid, *, refusal=False, kip=False, answer="Готовое how-to-описание шага.",
         task_ctx=True):
    """Синтетическая quality_snapshot-запись.

    refusal -> abstained.faithfulness True (is_refusal сработает по воздержанию);
    kip -> в context_ids появляется chunk:page:KIP-... (KIP дошёл в top-k).
    """
    ctx_ids = []
    if task_ctx:
        ctx_ids.append(f"chunk:task:KAFKA-{sid}#0")
    if kip:
        ctx_ids.append(f"chunk:page:KIP-{sid}#0")
    return {
        "source_id": f"task:KAFKA-{sid}",
        "question": f"как сделать {sid}?",
        "route": "howto",
        "answer": answer,
        "grounded": not refusal,
        "context_ids": ctx_ids,
        "context_texts": ["ctx"],
        "metrics": {"faithfulness": 1.0},
        "abstained": {"faithfulness": bool(refusal)},
    }


def test_unmatched_unanswered_flags_denominator_shrink():
    # ранее-безответный "9" отсутствует в AFTER -> усадка знаменателя, должна считаться
    before = [_rec("1", refusal=True), _rec("9", refusal=True)]
    after = [_rec("1", kip=True)]  # "9" пропал
    assert unmatched_unanswered(before, after) == 1
    # и отчёт несёт предупреждение
    rep = report(before, after)
    assert rep["unmatched_unanswered"] == 1


def test_join_dedups_duplicate_before_keys():
    # дубликат ключа в BEFORE не должен удваивать знаменатель
    before = [_rec("1", refusal=True), _rec("1", refusal=True)]
    after = [_rec("1", kip=True)]
    assert primary_lift(before, after)["denominator"] == 1


def test_kip_in_topk_detects_page_chunk():
    assert kip_in_topk(_rec("1", kip=True)) is True
    assert kip_in_topk(_rec("1", kip=False)) is False


def test_classify_transition_all_four():
    assert classify_transition(_rec("1", refusal=True), _rec("1")) == "refusal_to_answered"
    assert classify_transition(_rec("1"), _rec("1", refusal=True)) == "answered_to_refusal"
    assert classify_transition(_rec("1"), _rec("1")) == "stayed_answered"
    assert (classify_transition(_rec("1", refusal=True), _rec("1", refusal=True))
            == "stayed_refusal")


def test_happy_refusal_to_answered_with_kip():
    """refusal(before)->answered(after) c chunk:page: -> в лифте И в KIP-подмножестве."""
    before = [_rec("1", refusal=True)]
    after = [_rec("1", kip=True)]
    lift = primary_lift(before, after)
    assert lift == {"numerator": 1, "denominator": 1, "rate": 1.0}
    diag = diagnostic_subset(before, after)
    assert diag["size"] == 1 and diag["became_answered"] == 1
    assert diag["no_kip_count"] == 0


def test_primary_denominator_is_all_unanswered_subset_smaller():
    """Первичный знаменатель = ВСЕ ранее-безответные; KIP-срез меньше со своим n."""
    before = [_rec("1", refusal=True), _rec("2", refusal=True), _rec("3", refusal=True)]
    # 1: answered с KIP; 2: answered БЕЗ KIP; 3: остался отказ.
    after = [_rec("1", kip=True), _rec("2", kip=False), _rec("3", refusal=True)]
    lift = primary_lift(before, after)
    assert lift["denominator"] == 3  # все ранее-безответные
    assert lift["numerator"] == 2    # 1 и 2 стали отвечены
    diag = diagnostic_subset(before, after)
    assert diag["previously_unanswered"] == 3
    assert diag["size"] == 1          # только #1 с KIP в top-k
    assert diag["became_answered"] == 1
    assert diag["no_kip_count"] == 2  # #2 и #3 без KIP
    assert diag["size"] < lift["denominator"]


def test_r7_reachability_gate_no_kip_flags_retrieval():
    """R7: ни одного chunk:page: -> вердикт про retrieval / not genre, не genre-negative."""
    after_over_unanswered = [_rec("1", kip=False), _rec("2", kip=False)]
    verdict = reachability_gate(after_over_unanswered)
    assert "retrieval" in verdict
    assert "not genre" in verdict


def test_r7_reachable_when_kip_present():
    verdict = reachability_gate([_rec("1", kip=True), _rec("2", kip=False)])
    assert "reachable" in verdict


def test_r8_noise_floor():
    """R8: лифт ниже флип-порога -> not ok; строго выше -> ok; равенство -> not ok."""
    assert noise_floor_ok(observed_lift_count=2, flip_rate_count=5) is False
    assert noise_floor_ok(observed_lift_count=6, flip_rate_count=5) is True
    assert noise_floor_ok(observed_lift_count=5, flip_rate_count=5) is False


def test_r9_displacement_lists_dropped_faith():
    """R9: stayed_answered с упавшей Qwen-faithfulness попадает в список."""
    before = [{"source_id": "task:KAFKA-1", "question": "q1", "answer": "ok",
               "abstained": {"faithfulness": False}, "qwen_faithfulness": 0.9}]
    after = [{"source_id": "task:KAFKA-1", "question": "q1", "answer": "ok",
              "abstained": {"faithfulness": False}, "qwen_faithfulness": 0.4}]
    disp = displacement_check(before, after,
                              faith_before_key=lambda r: r.get("qwen_faithfulness"),
                              faith_after_key=lambda r: r.get("qwen_faithfulness"))
    assert len(disp) == 1
    assert disp[0]["source_id"] == "task:KAFKA-1"
    assert disp[0]["drop"] == 0.5


def test_r9_no_displacement_when_faith_holds():
    before = [{"source_id": "task:KAFKA-1", "question": "q1", "answer": "ok",
               "abstained": {"faithfulness": False}, "qwen_faithfulness": 0.5}]
    after = [{"source_id": "task:KAFKA-1", "question": "q1", "answer": "ok",
              "abstained": {"faithfulness": False}, "qwen_faithfulness": 0.7}]
    disp = displacement_check(before, after,
                              faith_before_key=lambda r: r.get("qwen_faithfulness"),
                              faith_after_key=lambda r: r.get("qwen_faithfulness"))
    assert disp == []


def test_edge_answered_to_refusal_not_counted_as_lift():
    """answered(before)->refusal(after): классифицирован, НЕ в лифте."""
    before = [_rec("1")]                    # отвечен
    after = [_rec("1", refusal=True)]       # стал отказом
    assert classify_transition(before[0], after[0]) == "answered_to_refusal"
    lift = primary_lift(before, after)
    assert lift["denominator"] == 0  # не был ранее-безответным
    assert lift["numerator"] == 0


def test_attach_qwen_faith_pure_merge():
    """attach_qwen_faith джойнит по (source_id, question), не мутирует вход."""
    records = [{"source_id": "task:KAFKA-1", "question": "q1"}]
    cross = [{"source_id": "task:KAFKA-1", "question": "q1",
              "qwen": {"faithfulness": 0.8}, "qwen_faith_abstained": False}]
    merged = attach_qwen_faith(records, cross)
    assert merged[0]["qwen_faithfulness"] == 0.8
    assert "qwen_faithfulness" not in records[0]  # вход не тронут


def test_report_assembles_all_guards():
    """report() собирает лифт, срез, R7/R8/R9 и примеры детерминированно."""
    before = [_rec("1", refusal=True), _rec("2", refusal=True)]
    after = [_rec("1", kip=True), _rec("2", kip=False)]
    rep = report(before, after, flip_rate_count=0)
    assert rep["primary_lift"]["denominator"] == 2
    assert rep["diagnostic"]["size"] == 1
    assert "reachable" in rep["r7"]              # #1 донёс KIP
    assert rep["r8"]["ok"] is True               # лифт 2 > флип 0
    assert rep["r9"]["regressions"] == []        # faith не задана -> нет регресса
    assert len(rep["examples"]) == 1             # только #1 refusal->answered+KIP
    assert rep["examples"][0]["kip_id"].startswith("chunk:page:")


def test_report_r7_retrieval_failure_when_no_kip():
    before = [_rec("1", refusal=True), _rec("2", refusal=True)]
    after = [_rec("1", kip=False), _rec("2", kip=False)]
    rep = report(before, after, flip_rate_count=0)
    assert "retrieval" in rep["r7"] and "not genre" in rep["r7"]
    assert rep["diagnostic"]["size"] == 0
