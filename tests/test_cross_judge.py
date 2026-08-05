"""cross_judge: независимый судья оценивает сохранённые (ответ, context_texts)."""

from __future__ import annotations

from graphrag.llm.base import LLMClient

from eval.cross_judge import cross_judge_records, rejudge_failures

_SCORES = {"faithfulness": 0.4, "answer_relevance": 0.3, "context_precision": 0.6}


class FakeJudge(LLMClient):
    """Судья: на любой judge-промпт возвращает свой JSON-балл."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        for key, value in _SCORES.items():
            if f'"{key}"' in prompt:
                return f'{{"{key}": {value}}}'
        return "{}"


def _rec(**kw):
    base = dict(
        source_id="task:1", question="как починить?", route="mixed", grounded=True,
        abstained={"faithfulness": False},
        answer="Ответ по существу.", context_texts=["фрагмент контекста про clients"],
        metrics={"faithfulness": 1.0, "answer_relevance": 0.9, "context_precision": 0.8},
    )
    base.update(kw)
    return base


def test_cross_judge_puts_qwen_beside_deepseek():
    out = cross_judge_records([_rec()], FakeJudge("x"))
    assert len(out) == 1
    r = out[0]
    # DeepSeek-оценки из снимка сохранены.
    assert r["deepseek"]["faithfulness"] == 1.0
    # Независимый судья проставил свои.
    assert r["qwen"]["faithfulness"] == 0.4
    assert r["qwen"]["answer_relevance"] == 0.3
    assert r["qwen"]["context_precision"] == 0.6
    assert r["source_id"] == "task:1"


def test_cross_judge_uses_stored_context_not_refetch():
    """Судья видит сохранённый context_texts (пустой контекст -> судья всё равно
    вызывается на нём, а не рефетчит по id)."""
    out = cross_judge_records([_rec(context_texts=[])], FakeJudge("x"))
    # precision-судья вызван на пустом контексте и вернул балл (не упал, не рефетчил).
    assert out[0]["qwen"]["context_precision"] == 0.6


def test_cross_judge_empty_records():
    assert cross_judge_records([], FakeJudge("x")) == []


class FlakyFaithJudge(LLMClient):
    """faith сбоит на первом вызове (пустой JSON -> None), потом успешен.

    Модель транзиентного сбоя (сеть/rate-limit): второй вызов уже валиден.
    relevance/precision всегда валидны.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.faith_calls = 0

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        if '"faithfulness"' in prompt:
            self.faith_calls += 1
            return "{}" if self.faith_calls == 1 else '{"faithfulness": 0.9}'
        if '"answer_relevance"' in prompt:
            return '{"answer_relevance": 0.3}'
        return '{"context_precision": 0.6}'


def test_retry_recovers_transient_faith_failure():
    """faith=None БЕЗ воздержания на первом вызове -> ретрай заполняет."""
    judge = FlakyFaithJudge("x")
    out = cross_judge_records([_rec()], judge)
    assert out[0]["qwen"]["faithfulness"] == 0.9
    assert out[0]["qwen_faith_abstained"] is False
    assert judge.faith_calls == 2  # был ретрай


def test_rejudge_failures_fills_only_failures():
    """Дозаполняет faith=None&не-воздержание из снимка; воздержания не трогает."""
    cross = [
        {"source_id": "task:1", "question": "q1",
         "qwen": {"faithfulness": None}, "qwen_faith_abstained": False},   # сбой -> чинится
        {"source_id": "task:2", "question": "q2",
         "qwen": {"faithfulness": None}, "qwen_faith_abstained": True},    # воздержание -> не трогать
        {"source_id": "task:3", "question": "q3",
         "qwen": {"faithfulness": 0.5}, "qwen_faith_abstained": False},    # уже есть -> не трогать
    ]
    snapshot = [
        {"source_id": "task:1", "question": "q1", "answer": "a", "context_texts": ["c"]},
    ]
    residual = rejudge_failures(cross, snapshot, FlakyFaithJudge("x"))
    assert residual == 0
    assert cross[0]["qwen"]["faithfulness"] == 0.9   # заполнено ретраем
    assert cross[1]["qwen"]["faithfulness"] is None   # воздержание нетронуто
    assert cross[1]["qwen_faith_abstained"] is True
    assert cross[2]["qwen"]["faithfulness"] == 0.5    # существующее нетронуто
