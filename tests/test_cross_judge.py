"""cross_judge: независимый судья оценивает сохранённые (ответ, context_texts)."""

from __future__ import annotations

from graphrag.llm.base import LLMClient

from eval.cross_judge import cross_judge_records

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
