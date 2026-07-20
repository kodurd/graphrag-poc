"""Метрики: retrieval (precision/recall/F1, recall@k), точность рёбер графа,
faithfulness генерации (LLM-as-judge). Чистые функции + опциональный judge.
"""

from __future__ import annotations

from graphrag.llm.base import LLMClient


def precision_recall_f1(retrieved: list[str] | set[str], expected: list[str] | set[str]) -> dict:
    r, e = set(retrieved), set(expected)
    tp = len(r & e)
    precision = tp / len(r) if r else 0.0
    recall = tp / len(e) if e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def recall_at_k(ranked_ids: list[str], expected: list[str] | set[str], k: int) -> float:
    e = set(expected)
    if not e:
        return 0.0
    return len(set(ranked_ids[:k]) & e) / len(e)


def edge_precision_recall(
    predicted: list[tuple], gold: list[tuple]
) -> dict:
    """Точность извлечённых рёбер. Рёбра — кортежи (from, type, to)."""
    return precision_recall_f1({tuple(x) for x in predicted}, {tuple(x) for x in gold})


def candidate_entity_id(candidate_id: str) -> str:
    """Сводит id кандидата к id сущности: chunk:task:KAFKA-1#0 -> task:KAFKA-1."""
    if candidate_id.startswith("chunk:"):
        inner = candidate_id[len("chunk:") :]
        return inner.rsplit("#", 1)[0]
    return candidate_id


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_item: list[dict], keys: tuple[str, ...] = ("precision", "recall", "f1")) -> dict:
    return {k: mean([d[k] for d in per_item]) for k in keys}


_FAITH_PROMPT = (
    "Оцени ВЕРНОСТЬ ОТВЕТА КОНТЕКСТУ: долю проверяемых фактических утверждений "
    "ОТВЕТА, подтверждённых КОНТЕКСТОМ (0..1).\n"
    "ВАЖНО: оценивай ТОЛЬКО сами утверждения, игнорируя форму. Фразы вроде "
    "«на основе контекста невозможно описать X», «детали не раскрыты», "
    "«без логов нельзя» — это НЕ утверждения и НЕ ошибки; они не понижают балл. "
    "Считай только позитивные фактические утверждения и проверяй КАЖДОЕ по "
    "контексту.\n"
    "Пример: ответ «Конкретную реализацию описать невозможно, но в контексте "
    "сказано, что методы реализуются через генератор, а не хардкодом» содержит "
    "ОДНО утверждение (про генератор); если оно есть в контексте — faithfulness "
    "близок к 1.0, а не 0 (оговорка «невозможно» не в счёт).\n"
    "Если позитивных фактических утверждений НЕТ вовсе (только оговорка/отказ) — "
    'верни {"faithfulness": null, "abstained": true}.\n'
    'Иначе верни {"faithfulness": <число 0..1>}. Только JSON.\n\n'
)


def judge_faithfulness(
    llm: LLMClient, answer: str, context_texts: list[str]
) -> tuple[float | None, bool]:
    """LLM-as-judge: (score, abstained).

    - score — доля подтверждённых контекстом утверждений (0..1), либо None.
    - abstained — True, когда ответ не содержит проверяемых утверждений (чистое
      воздержание): score тогда None, но это НЕ сбой судьи.
    - Сбой судьи (сеть/невалидный JSON/нет ключа) → (None, False): None отличает
      отказ оценки от честного нуля, а False — воздержание от сбоя.

    Разбирает JSON сам (не через `_judge_score`), т.к. несёт флаг `abstained`;
    остальные 4 судьи и `_judge_score` не затрагиваются.
    """
    ctx = "\n".join(f"- {t}" for t in context_texts)
    prompt = f"{_FAITH_PROMPT}КОНТЕКСТ:\n{ctx}\n\nОТВЕТ:\n{answer}"
    try:
        data = llm.extract_json(prompt)
        if not isinstance(data, dict):
            return None, False
        if data.get("abstained") is True:
            return None, True
        if "faithfulness" not in data:
            return None, False
        val = data["faithfulness"]
        if val is None:  # явный null — тоже воздержание
            return None, True
        return max(0.0, min(1.0, float(val))), False
    except Exception:
        return None, False


def _judge_score(llm: LLMClient, prompt: str, key: str) -> float | None:
    """Общий каркас LLM-судьи: промпт -> число 0..1 либо None при сбое.

    None (а не 0.0) при сбое/невалидном JSON — чтобы отказ оценки отличался от
    честного нуля и не занижал агрегат.
    """
    try:
        data = llm.extract_json(prompt)
        if not isinstance(data, dict) or key not in data:
            return None
        return max(0.0, min(1.0, float(data[key])))
    except Exception:
        return None


# --- Reference-free судьи: эталон не нужен ------------------------------------

_RELEVANCE_PROMPT = (
    "Оцени, насколько ОТВЕТ отвечает по существу на ВОПРОС (а не «рядом»). "
    'Верни JSON {"answer_relevance": <число 0..1>}. Только JSON.\n\n'
)

_CTX_PRECISION_PROMPT = (
    "Оцени, какая доля фрагментов КОНТЕКСТА релевантна ВОПРОСУ. "
    'Верни JSON {"context_precision": <число 0..1>}. Только JSON.\n\n'
)


def judge_answer_relevance(llm: LLMClient, question: str, answer: str) -> float | None:
    """LLM-as-judge: отвечает ли ответ на вопрос по существу (0..1)."""
    prompt = f"{_RELEVANCE_PROMPT}ВОПРОС:\n{question}\n\nОТВЕТ:\n{answer}"
    return _judge_score(llm, prompt, "answer_relevance")


def judge_context_precision(
    llm: LLMClient, question: str, context_texts: list[str]
) -> float | None:
    """LLM-as-judge: доля релевантных вопросу фрагментов контекста (0..1)."""
    ctx = "\n".join(f"- {t}" for t in context_texts)
    prompt = f"{_CTX_PRECISION_PROMPT}ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{ctx}"
    return _judge_score(llm, prompt, "context_precision")


# --- Reference-required судьи: нужен эталонный ответ --------------------------

_CORRECTNESS_PROMPT = (
    "Оцени, насколько ОТВЕТ согласуется с ЭТАЛОНОМ по существу вопроса "
    "(расхождения в формулировках не штрафуем, фактические — штрафуем). "
    'Верни JSON {"answer_correctness": <число 0..1>}. Только JSON.\n\n'
)

_CTX_RECALL_PROMPT = (
    "Оцени, какая доля утверждений ЭТАЛОНА покрыта КОНТЕКСТОМ. "
    'Верни JSON {"context_recall": <число 0..1>}. Только JSON.\n\n'
)


def judge_answer_correctness(
    llm: LLMClient, question: str, answer: str, reference: str
) -> float | None:
    """LLM-as-judge: согласованность ответа с эталоном (0..1)."""
    prompt = (
        f"{_CORRECTNESS_PROMPT}ВОПРОС:\n{question}\n\n"
        f"ЭТАЛОН:\n{reference}\n\nОТВЕТ:\n{answer}"
    )
    return _judge_score(llm, prompt, "answer_correctness")


def judge_context_recall(
    llm: LLMClient, reference: str, context_texts: list[str]
) -> float | None:
    """LLM-as-judge: доля утверждений эталона, покрытых контекстом (0..1)."""
    ctx = "\n".join(f"- {t}" for t in context_texts)
    prompt = f"{_CTX_RECALL_PROMPT}ЭТАЛОН:\n{reference}\n\nКОНТЕКСТ:\n{ctx}"
    return _judge_score(llm, prompt, "context_recall")
