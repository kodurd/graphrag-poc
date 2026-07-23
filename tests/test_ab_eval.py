"""U2: парный A/B lexical vs cross-encoder — чистые функции дельт/агрегатов/вердикта."""

from __future__ import annotations

import pytest

from graphrag.llm.base import LLMClient

from eval.ab_eval import (
    answered_flips,
    changed_ranking,
    flip_correctness,
    is_answered,
    multihop_subset,
    n_needed,
    paired_deltas,
    per_version_summary,
    permutation_pvalue,
    render_ab_report,
    render_changed_ranking,
    render_multihop_ab_report,
    run_ab_eval,
    triage_verdict,
    version_abstention,
)


def _rrec(context_ids, answer="ответ"):
    return {"context_ids": context_ids, "answer": answer,
            "abstained": {"faithfulness": False}, "metrics": {}}


def _rec(*, abstained=False, **metrics):
    return {"abstained": {"faithfulness": abstained}, "metrics": metrics}


def _pair(question, lex, ce):
    return {"question": question, "lexical": lex, "cross_encoder": ce}


def _mrec(route="multihop", *, abstained=False, **metrics):
    return {"route": route, "abstained": {"faithfulness": abstained}, "metrics": metrics}


# --- U-multihop: изоляция multihop-подмножества + flip'ы воздержание→ответ + корректность ---

def test_multihop_subset_selects_only_multihop():
    pairs = [_pair("q1", _mrec("multihop"), _mrec("multihop")),
             _pair("q2", _mrec("mixed"), _mrec("mixed"))]
    assert [p["question"] for p in multihop_subset(pairs)] == ["q1"]


def test_answered_flips_before_abstained_after_answered():
    # arm_off=lexical (graph-only/before), arm_on=cross_encoder (full/after)
    pairs = [
        _pair("flip", _mrec(abstained=True), _mrec(abstained=False)),    # off возд, on отв
        _pair("both_ans", _mrec(abstained=False), _mrec(abstained=False)),
        _pair("reverse", _mrec(abstained=False), _mrec(abstained=True)),  # обратный — не flip
    ]
    assert [p["question"] for p in answered_flips(pairs)] == ["flip"]


def test_flip_correctness_mean_of_after_answers():
    pairs = [
        _pair("f1", _mrec(abstained=True), _mrec(abstained=False, faithfulness=0.8)),
        _pair("f2", _mrec(abstained=True), _mrec(abstained=False, faithfulness=0.6)),
        _pair("noflip", _mrec(abstained=False), _mrec(abstained=False, faithfulness=0.1)),
    ]
    c = flip_correctness(pairs, "faithfulness")
    assert c["n_flips"] == 2 and c["mean"] == pytest.approx(0.7)  # только after-ответы flip'ов


def test_flip_correctness_no_flips():
    pairs = [_pair("q", _mrec(abstained=False), _mrec(abstained=False, faithfulness=0.9))]
    c = flip_correctness(pairs, "faithfulness")
    assert c["n_flips"] == 0 and c["mean"] is None


def test_render_multihop_report_labels_arms_and_metrics():
    pairs = [_pair("q1", _mrec("multihop", abstained=True),
                   _mrec("multihop", abstained=False, faithfulness=0.7, context_precision=0.6))]
    md = render_multihop_ab_report(pairs, ["faithfulness", "context_precision"])
    assert "graph-only" in md and "full" in md  # честные метки плеч, не lexical/cross-encoder
    assert "fusion" not in md.lower()  # не переиспользован fusion-словесный рендер
    assert "faithfulness" in md


# --- is_answered ---

def test_is_answered_reads_abstention_flag():
    assert is_answered(_rec(abstained=False)) is True
    assert is_answered(_rec(abstained=True)) is False


# --- дельты только на совместно-отвечённых ---

def test_paired_deltas_only_jointly_answered():
    pairs = [
        _pair("both", _rec(answer_correctness=0.6), _rec(answer_correctness=0.9)),   # обе -> дельта
        _pair("lex_abst", _rec(abstained=True), _rec(answer_correctness=0.8)),       # lexical воздержался
        _pair("ce_none", _rec(answer_correctness=0.5), _rec(answer_correctness=None)),  # метрика None
    ]
    d = paired_deltas(pairs, "answer_correctness")
    assert [x["question"] for x in d] == ["both"]
    assert d[0]["delta"] == pytest.approx(0.3)


# --- пер-версийный агрегат по собственному множеству отвечённых ---

def test_per_version_summary_uses_own_answered_set():
    pairs = [
        _pair("q1", _rec(abstained=True), _rec(context_precision=0.8)),   # ответил только CE
        _pair("q2", _rec(context_precision=0.4), _rec(context_precision=0.6)),
    ]
    lex = per_version_summary(pairs, "lexical", "context_precision")
    ce = per_version_summary(pairs, "cross_encoder", "context_precision")
    assert lex["n"] == 1 and lex["mean"] == pytest.approx(0.4)           # q1 исключён (воздержался)
    assert ce["n"] == 2 and ce["mean"] == pytest.approx(0.7)             # q1 входит в CE


# --- воздержания по версиям ---

def test_version_abstention_rate():
    pairs = [
        _pair("q1", _rec(abstained=True), _rec()),
        _pair("q2", _rec(abstained=True), _rec()),
        _pair("q3", _rec(), _rec(abstained=True)),
    ]
    assert version_abstention(pairs, "lexical")["rate"] == pytest.approx(2 / 3)
    assert version_abstention(pairs, "cross_encoder")["rate"] == pytest.approx(1 / 3)


# --- перестановочный тест + триаж-вердикт (U1) ---

def test_permutation_all_positive_significant():
    # все дельты одного знака -> знаковые развороты почти не воспроизводят наблюдённый эффект
    assert permutation_pvalue([0.2] * 8) < 0.05


def test_permutation_symmetric_not_significant():
    assert permutation_pvalue([0.2, -0.2, 0.2, -0.2, 0.2, -0.2]) > 0.05


def test_triage_verdict_b_better_on_positive():
    # соглашение: delta = B - A; delta>0 значимо -> B лучше
    v = triage_verdict([{"delta": 0.2}] * 8)
    assert v["verdict"] == "B_better" and v["p"] < 0.05 and v["n_needed"] is None


def test_triage_verdict_a_better_on_negative():
    v = triage_verdict([{"delta": -0.2}] * 8)
    assert v["verdict"] == "A_better"


def test_triage_verdict_inconclusive_sets_n_needed():
    # малый ненулевой эффект при n=4 -> не значим, но n_needed оценивается
    v = triage_verdict([{"delta": 0.1}, {"delta": -0.05}, {"delta": 0.08}, {"delta": -0.02}])
    assert v["verdict"] == "inconclusive" and v["n_needed"] is not None


def test_triage_verdict_outlier_robust_at_small_n():
    # один выброс не делает результат значимым (знаковый тест устойчив к величине)
    v = triage_verdict([{"delta": 0.0}] * 6 + [{"delta": 0.9}])
    assert v["verdict"] == "inconclusive"


def test_n_needed_grows_as_effect_shrinks():
    assert n_needed(0.1, sd=0.2) > n_needed(0.3, sd=0.2)


def test_permutation_exact_and_sample_paths_reproducible():
    small = [0.1, -0.05, 0.2, 0.0]                      # n=4 -> точный перебор
    big = [0.1] * 25                                     # n=25 -> сэмпл
    for vals in (small, big):
        assert 0.0 <= permutation_pvalue(vals) <= 1.0
    # сэмпл-путь воспроизводим при том же сиде
    assert permutation_pvalue(big, seed=1) == permutation_pvalue(big, seed=1)


# --- раннер: две записи на вопрос через инъекцию evaluate_fn ---

def test_run_ab_eval_two_records_per_question():
    calls = []

    def fake_eval(retr, llm, question, *, reference=None, source_id=None):
        calls.append(retr)
        return _rec(answer_correctness=0.5)

    pairs = run_ab_eval(
        "RETR_LEX", "RETR_CE", llm=None,
        questions=[{"question": "q1", "source_id": "task:A"}],
        labeled=[{"question": "q2", "reference": "эталон", "source_id": "task:B"}],
        evaluate_fn=fake_eval,
    )
    assert len(pairs) == 2
    assert set(pairs[0]) == {"question", "lexical", "cross_encoder"}
    assert calls == ["RETR_LEX", "RETR_CE", "RETR_LEX", "RETR_CE"]  # обе версии на каждый вопрос


# --- рендер отчёта (R6) ---

def test_render_ab_report_shows_versions_abstention_verdict():
    pairs = [
        _pair("q1", _rec(context_precision=0.4), _rec(context_precision=0.7)),
        _pair("q2", _rec(abstained=True), _rec(context_precision=0.6)),
    ]
    md = render_ab_report(pairs, ["context_precision"])
    assert "Воздержания: lexical 1/2" in md and "cross-encoder 0/2" in md
    assert "lexical: n=1" in md and "cross-encoder: n=2" in md  # пер-версийные множества
    assert "неразрешимо" in md  # 1 совместный вопрос -> перестановочный тест неразрешим
    assert "+0.30" in md  # таблица дельт


# --- U3: сверка изменившегося ранжирования ---

def test_changed_ranking_selects_only_differing():
    pairs = [
        _pair("same", _rrec(["a", "b"]), _rrec(["a", "b"])),   # идентичны -> исключён
        _pair("diff", _rrec(["a", "b"]), _rrec(["b", "a"])),   # порядок иной -> включён
    ]
    assert [p["question"] for p in changed_ranking(pairs)] == ["diff"]


def test_render_changed_ranking_contains_both_sides():
    pairs = [_pair("q", _rrec(["a"], "лекс-ответ"), _rrec(["b"], "це-ответ"))]
    md = render_changed_ranking(pairs)
    assert "лекс-ответ" in md and "це-ответ" in md
    assert "['a']" in md and "['b']" in md
    assert "смещение судьи" in md  # caveat на месте


def test_render_changed_ranking_empty_case():
    pairs = [_pair("same", _rrec(["a"]), _rrec(["a"]))]
    md = render_changed_ranking(pairs)
    assert "Изменений ранжирования нет." in md
    assert "Чек-лист аудита формы" in md  # аудит-шапка есть даже при отсутствии изменений


# --- U2: инвариант «плечо не в judge-входе» + аудит формы ---

class _RecordingLLM(LLMClient):
    """Записывает все промпты; судьям отдаёт JSON, генерации — текст с цитатой."""

    def __init__(self):
        super().__init__("x")
        self.prompts: list[str] = []

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        for key in ("faithfulness", "answer_relevance", "context_precision",
                    "answer_correctness", "context_recall"):
            if f'"{key}"' in prompt:
                return f'{{"{key}": 0.5}}'
        return "Ответ по существу [источник: https://issues/KAFKA-1]"


class _FakeRetriever:
    def __init__(self, candidates):
        self._c = candidates

    def retrieve(self, query):
        return {"route": "mixed", "candidates": self._c}


def test_judge_input_carries_no_arm_label():
    from eval.quality_eval import evaluate_question

    # два «плеча» с разными кандидатами; метка конфига (lexical/cross_encoder) не подаётся
    arms = [
        [{"id": "chunk:task:1#0", "text": "clients reconnect", "uri": "https://issues/KAFKA-1"}],
        [{"id": "chunk:task:2#0", "text": "streams rebalance", "uri": "https://issues/KAFKA-2"}],
    ]
    for cands in arms:
        llm = _RecordingLLM()
        evaluate_question(_FakeRetriever(cands), llm, "почему падает?", source_id="task:1")
        judge_prompts = [p for p in llm.prompts
                         if "Верни JSON" in p or "abstained" in p or '"faithfulness"' in p]
        assert judge_prompts, "судьи должны были вызваться"
        for p in judge_prompts:
            low = p.lower()
            assert "lexical" not in low and "cross_encoder" not in low


def test_changed_ranking_sheet_has_lengths_and_checklist():
    pairs = [_pair("q", _rrec(["a"], "коротко"), _rrec(["b"], "гораздо длиннее ответ тут"))]
    md = render_changed_ranking(pairs)
    assert "Чек-лист аудита формы" in md
    assert f"длина ответа: {len('коротко')}" in md
    assert f"длина ответа: {len('гораздо длиннее ответ тут')}" in md
