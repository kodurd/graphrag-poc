"""Прогонный харнесс качества — на фейковом ретривере и скриптованном LLM.

Без живого Neo4j: ретривер подменён фейком, LLM различает промпт генерации
(«Ответь по существу») и промпты судей («Верни JSON …»).
"""

from __future__ import annotations

from eval.quality_eval import evaluate_question, run_quality_eval
from graphrag.llm.base import LLMClient

URI = "https://issues.apache.org/jira/browse/KAFKA-1"

# Ключи судей -> значение, которое отдаёт скриптованный LLM.
_JUDGE_SCORES = {
    "faithfulness": 0.9,
    "answer_relevance": 0.8,
    "context_precision": 0.7,
    "answer_correctness": 0.6,
    "context_recall": 0.5,
}


class FakeRetriever:
    def __init__(self, candidates=None, route="mixed"):
        self._candidates = (
            [{"id": "chunk:task:1#0", "text": "clients reconnect loop", "uri": URI}]
            if candidates is None
            else candidates
        )
        self._route = route

    def retrieve(self, query):
        return {"route": self._route, "candidates": self._candidates}


class ScriptedLLM(LLMClient):
    """Отдаёт ответ с валидной цитатой на генерацию и JSON — на судей."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        for key, value in _JUDGE_SCORES.items():
            if f'"{key}"' in prompt:
                return f'{{"{key}": {value}}}'
        return f"Ответ по существу [источник: {URI}]"


class BadJudgeLLM(ScriptedLLM):
    """Генерация работает, все судьи ломаются."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        if "Верни JSON" in prompt:
            raise RuntimeError("судья упал")
        return f"Ответ по существу [источник: {URI}]"


# --- один вопрос ---

def test_evaluate_question_collects_reference_free_metrics():
    rec = evaluate_question(FakeRetriever(), ScriptedLLM("x"), "почему падает?", source_id="task:1")
    assert rec["route"] == "mixed"
    assert rec["source_id"] == "task:1"
    assert rec["grounded"] is True
    assert rec["citations"] == [URI]
    assert rec["context_ids"] == ["chunk:task:1#0"]
    assert rec["metrics"]["faithfulness"] == 0.9
    assert rec["metrics"]["answer_relevance"] == 0.8
    assert rec["metrics"]["context_precision"] == 0.7
    # Без эталона reference-required метрики не считаются.
    assert "answer_correctness" not in rec["metrics"]


def test_evaluate_question_with_reference_adds_required_metrics():
    rec = evaluate_question(
        FakeRetriever(), ScriptedLLM("x"), "почему падает?", reference="эталонный ответ"
    )
    assert rec["reference"] == "эталонный ответ"
    assert rec["metrics"]["answer_correctness"] == 0.6
    assert rec["metrics"]["context_recall"] == 0.5


# --- воздержание faithfulness ---

class AbstainFaithLLM(ScriptedLLM):
    """Судья faithfulness отдаёт воздержание; остальные — как обычно."""

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        if '"faithfulness"' in prompt:
            return '{"faithfulness": null, "abstained": true}'
        return super()._raw_complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)


def test_abstention_flag_lands_outside_metrics():
    rec = evaluate_question(FakeRetriever(), AbstainFaithLLM("x"), "вопрос?")
    # score воздержания -> None в metrics; флаг -> sibling-поле вне metrics
    assert rec["metrics"]["faithfulness"] is None
    assert rec["abstained"]["faithfulness"] is True
    # инвариант metrics не нарушен: только число/None, флаг снаружи
    assert all(v is None or isinstance(v, (int, float)) for v in rec["metrics"].values())
    assert "abstained" not in rec["metrics"]


# --- устойчивость ---

def test_judge_failure_yields_none_without_crashing_run():
    rec = evaluate_question(FakeRetriever(), BadJudgeLLM("x"), "вопрос?")
    # Прогон дошёл до конца, ответ есть, а метрики честно None (не 0.0).
    assert rec["answer"].startswith("Ответ по существу")
    assert all(v is None for v in rec["metrics"].values())
    # Сбой != воздержание: флаг воздержания False.
    assert rec["abstained"]["faithfulness"] is False


def test_empty_candidates_give_honest_answer_without_crash():
    rec = evaluate_question(FakeRetriever(candidates=[]), ScriptedLLM("x"), "вопрос?")
    assert rec["grounded"] is False
    assert rec["citations"] == []
    assert rec["context_ids"] == []
    assert "Недостаточно данных" in rec["answer"]


# --- полный прогон ---

def test_run_quality_eval_covers_questions_and_labeled_slice():
    questions = [
        {"question": "вопрос один", "source_id": "task:1"},
        {"question": "вопрос два", "source_id": "task:2"},
    ]
    labeled = [{"question": "вопрос три", "reference": "эталон", "source_id": "task:3"}]

    result = run_quality_eval(FakeRetriever(), ScriptedLLM("x"), questions, labeled)

    assert result["counts"] == {"questions": 2, "labeled": 1, "total": 3}
    assert len(result["records"]) == 3
    # Только запись из размеченного среза несёт reference-required метрики.
    with_ref = [r for r in result["records"] if "reference" in r]
    assert len(with_ref) == 1
    assert with_ref[0]["metrics"]["answer_correctness"] == 0.6


def test_run_quality_eval_without_labeled_slice():
    result = run_quality_eval(FakeRetriever(), ScriptedLLM("x"), [{"question": "в?"}])
    assert result["counts"]["labeled"] == 0
    assert "answer_correctness" not in result["records"][0]["metrics"]


# --- прокидка faithfulness-сэмплов (config -> judge) ---

class CountingFaithLLM(ScriptedLLM):
    """Считает вызовы faithfulness-судьи и пишет судья-temperature."""

    def __init__(self, tag="x"):
        super().__init__(tag)
        self.faith_calls = 0
        self.faith_temps: list = []

    def _raw_complete(self, prompt, *, system=None, temperature=None, max_tokens=None):
        if '"faithfulness"' in prompt:
            self.faith_calls += 1
            self.faith_temps.append(temperature)
        return super()._raw_complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)


def test_faithfulness_samples_thread_to_judge():
    llm = CountingFaithLLM()
    evaluate_question(FakeRetriever(), llm, "почему?", faithfulness_samples=3, faithfulness_temperature=0.3)
    assert llm.faith_calls == 3  # N сэмплов дошли до judge_faithfulness
    assert llm.faith_temps == [0.3, 0.3, 0.3]  # judge-temp на каждый сэмпл


def test_default_single_faithfulness_call():
    llm = CountingFaithLLM()
    evaluate_question(FakeRetriever(), llm, "почему?")
    assert llm.faith_calls == 1  # дефолт N=1 (обратная совместимость)


def test_run_quality_eval_threads_samples():
    llm = CountingFaithLLM()
    run_quality_eval(FakeRetriever(), llm, [{"question": "в1?"}, {"question": "в2?"}],
                     faithfulness_samples=2)
    assert llm.faith_calls == 4  # 2 вопроса × 2 сэмпла
