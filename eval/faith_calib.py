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


def sample_variance(scores: list[float | None]) -> float:
    """Популяционная дисперсия скоров судьи по одному элементу (величина ШУМА).

    None-сэмплы (сбой/воздержание) игнорируются; < 2 скоров → 0 (дисперсии нет).
    """
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def diagnose(
    variance: float, residual: float, *,
    var_threshold: float = 0.05, residual_threshold: float = 0.2,
) -> dict:
    """Классифицирует ошибку судьи: `noise` / `bias` / `mixed` / `ok`.

    `variance` — средняя per-item дисперсия сэмплов (шум); `residual` — направленный
    остаток `mean(judge − human)` (смещение). Высокая дисперсия + малый |остаток| →
    шум (лечится агрегацией); малая дисперсия + большой |остаток| → смещение (агрегация
    НЕ поможет, нужна рубрика/decompose-verify); оба велики → mixed; оба малы → судья ok.
    """
    hi_var = variance >= var_threshold
    hi_res = abs(residual) >= residual_threshold
    if hi_var and hi_res:
        verdict = "mixed"
    elif hi_var:
        verdict = "noise"
    elif hi_res:
        verdict = "bias"
    else:
        verdict = "ok"
    return {"verdict": verdict, "variance": variance, "residual": residual}


def diagnose_run(baseline_pairs: list[dict], sampled_scores: list[list[float | None]]) -> dict:
    """Сводит замеры в диагноз. `baseline_pairs` — судья при temp=0 (для направленного
    остатка = смещение при отсутствии шума); `sampled_scores` — списки N сэмплов на элемент
    при judge-temp (для дисперсии = шум). Средняя per-item дисперсия + остаток → `diagnose`.
    """
    agr = judge_agreement(baseline_pairs)
    residual = agr["directional_residual"] or 0.0
    per_item_var = [sample_variance(s) for s in sampled_scores]
    mean_var = sum(per_item_var) / len(per_item_var) if per_item_var else 0.0
    return {**diagnose(mean_var, residual), "baseline_agreement": agr}
