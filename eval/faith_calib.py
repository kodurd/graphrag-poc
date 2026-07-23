"""Калибровка faithfulness-судьи против доверяемого ручного gold.

Инструмент-first: сначала мерим ошибку судьи (величину И направление) на
сбалансированном валидированном gold, потом диагностируем шум/смещение, потом
выбираем фикс. Чистые метрики — здесь; онлайн-прогон судьи — в раннере ниже.
"""

from __future__ import annotations

_BUCKETS = (0.0, 0.5, 1.0)


def _bucket(x: float) -> float:
    """Ближайший из {0, 0.5, 1} — судья двухполюсный, грубых бакетов хватает."""
    return min(_BUCKETS, key=lambda b: abs(x - b))


def judge_agreement(pairs: list[dict]) -> dict:
    """Согласие судьи с ручными метками. Каждый pair: `{human, judge, abstained}`.

    Оценённые — где `not abstained` и `judge`/`human` не None. Возвращает:
    - `mae` — средний `|judge − human|` (величина ошибки);
    - `directional_residual` — средний `judge − human` (ЗНАК ошибки: <0 = судья
      систематически занижает → смещение, ≈0 при большом mae → симметричный шум);
    - `bucket_agreement` — доля совпадений бакета {0,0.5,1};
    - `per_item`, `n_scored`, `n_abstained`.
    """
    scored = [
        p for p in pairs
        if not p.get("abstained") and p.get("judge") is not None and p.get("human") is not None
    ]
    n = len(scored)
    per_item = [
        {"human": p["human"], "judge": p["judge"],
         "abs_err": abs(p["judge"] - p["human"]),
         "bucket_match": _bucket(p["judge"]) == _bucket(p["human"])}
        for p in scored
    ]
    if n == 0:
        return {"mae": None, "directional_residual": None, "bucket_agreement": None,
                "per_item": per_item, "n_scored": 0,
                "n_abstained": sum(1 for p in pairs if p.get("abstained"))}
    mae = sum(d["abs_err"] for d in per_item) / n
    residual = sum(p["judge"] - p["human"] for p in scored) / n
    bucket_agreement = sum(1 for d in per_item if d["bucket_match"]) / n
    return {"mae": mae, "directional_residual": residual, "bucket_agreement": bucket_agreement,
            "per_item": per_item, "n_scored": n,
            "n_abstained": sum(1 for p in pairs if p.get("abstained"))}


def run_gold_judge(gold_items: list[dict], judge_fn) -> list[dict]:
    """Прогоняет судью по gold, возвращает пары `{human, judge, abstained}` для `judge_agreement`.

    `judge_fn(answer, context_texts) -> (score|None, abstained)` инъектируется для тестов;
    в онлайне — обёртка над `judge_faithfulness(llm, ...)`. Контекст берётся ИНЛАЙН из gold
    (`context_text`), а не реконструируется по id.
    """
    pairs: list[dict] = []
    for it in gold_items:
        score, abstained = judge_fn(it["answer"], [it["context_text"]])
        pairs.append({"human": it["human_faithfulness"], "judge": score, "abstained": abstained})
    return pairs
