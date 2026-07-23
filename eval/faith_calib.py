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


def holistic_verdict(
    agreement: dict, *, mae_bar: float, residual_bar: float
) -> dict:
    """Вердикт о существующем ХОЛИСТИЧЕСКОМ судье на трудном подмножестве по АБСОЛЮТНОМУ порогу.

    Отвечает «0.33 реально или артефакт?» без нового механизма. Над single-run
    `judge_agreement` (mae + направленный остаток) — НЕ дублирует `diagnose` (тот над
    сэмпл-дисперсией). Исходы:
    - `degenerate` — mae=None (пустое/всё-воздержалось подмножество), не падение;
    - `artifact` — `mae ≤ mae_bar`: судья согласен с ручными метками → провал был артефактом;
    - `real_failure_bias` — `mae > mae_bar` и `|остаток| ≥ residual_bar`: систематическое
      отклонение (любого знака; архетип — судья занижает faithful) → рубрика/decompose-verify;
    - `real_failure_noise` — `mae > mae_bar` и малый остаток: симметричный шум.
    """
    mae = agreement.get("mae")
    residual = agreement.get("directional_residual")
    if mae is None:
        verdict = "degenerate"
    elif mae <= mae_bar:
        verdict = "artifact"
    elif residual is not None and abs(residual) >= residual_bar:
        verdict = "real_failure_bias"
    else:
        verdict = "real_failure_noise"
    return {"verdict": verdict, "mae": mae, "residual": residual}


def beats_baseline(candidate_mae: float | None, baseline_mae: float | None) -> bool:
    """Строго ли кандидат-фикс лучше baseline по mae. Явные None-проверки: `base=0.0`
    (идеальный baseline) — валидное значение, а не «отсутствует» (баг `0.0 or ...`)."""
    if candidate_mae is None or baseline_mae is None:
        return False
    return candidate_mae < baseline_mae


_GOLD = "eval/trial/faith_gold.json"
_REPORT = "eval/trial/faith_calib_report.md"


def main(n_samples: int = 5) -> None:
    """Онлайн: диагностировать судью на gold и сравнить кандидат-фикс с temp=0-baseline.

    Дёшево (gold мал, контекст инлайн — ни Neo4j, ни ретрива/генерации, только судья).
    Нужен LLM. Пишет отчёт: baseline mae/residual, вердикт noise/bias/mixed, mae кандидата
    (среднее N сэмплов) — коммитить фикс только если он бьёт baseline.
    """
    import json

    from graphrag.config import load_settings
    from graphrag.llm import build_llm

    from eval.metrics import _judge_faithfulness_once, judge_faithfulness

    s = load_settings()
    gold = json.load(open(_GOLD, encoding="utf-8"))["items"]
    llm = build_llm(s.llm, role="generation")
    temp = s.eval.faithfulness_judge_temperature

    # temp=0 baseline (noise-free): направленный остаток = смещение
    baseline_pairs = run_gold_judge(
        gold, lambda a, c: judge_faithfulness(llm, a, c, n_samples=1, temperature=0.0))
    # N сэмплов при judge-temp: дисперсия = шум
    sampled_scores = [
        [_judge_faithfulness_once(llm, it["answer"], [it["context_text"]], temperature=temp)[0]
         for _ in range(n_samples)]
        for it in gold
    ]
    diag = diagnose_run(baseline_pairs, sampled_scores)
    base = diag["baseline_agreement"]

    # кандидат-фикс (noise-ветка): среднее N сэмплов
    cand_pairs = run_gold_judge(
        gold, lambda a, c: judge_faithfulness(llm, a, c, n_samples=n_samples, temperature=temp))
    cand = judge_agreement(cand_pairs)

    def _f(x):
        return "—" if x is None else f"{x:.3f}"

    lines = [
        "# Калибровка faithfulness-судьи",
        "",
        f"Gold: {len(gold)} записей (сбалансирован, метки по per-claim entailment).",
        "",
        f"**Диагноз: {diag['verdict']}** (дисперсия={_f(diag['variance'])}, "
        f"направленный остаток={_f(diag['residual'])}).",
        f"- temp=0 baseline: mae={_f(base['mae'])}, bucket-согласие={_f(base['bucket_agreement'])}",
        f"- кандидат (среднее {n_samples} сэмплов): mae={_f(cand['mae'])}, "
        f"bucket-согласие={_f(cand['bucket_agreement'])}",
        "",
        ("Кандидат БЬЁТ baseline — фикс оправдан." if beats_baseline(cand["mae"], base["mae"])
         else "Кандидат НЕ бьёт baseline — сэмплинг не оправдан; при 'bias' нужна рубрика/decompose-verify."),
        "",
        "⚠️ n мал, how-to-архетип — вывод направленный, не статзначимый; метки посеяны агентом.",
    ]
    open(_REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"DONE diagnosis={diag['verdict']} baseline_mae={_f(base['mae'])} "
          f"candidate_mae={_f(cand['mae'])} -> {_REPORT}")


_CONFIRM_REPORT = "eval/trial/faith_confirm_report.md"


def main_confirm(mae_bar: float = 0.25, residual_bar: float = 0.2) -> None:
    """Онлайн: пере-прогон СУЩЕСТВУЮЩЕГО холистического судьи на ТРУДНОМ подмножестве gold
    и вердикт по абсолютному порогу — реально ли судья проваливается или 0.33 был артефактом.

    Дёшево (только hard-подмножество, контекст инлайн — ни Neo4j, ни ретрива). Пороги
    `mae_bar`/`residual_bar` номинальные — калибруются по первому прогону (Open Question плана).
    """
    import json

    from graphrag.config import load_settings
    from graphrag.llm import build_llm

    from eval.metrics import judge_faithfulness

    s = load_settings()
    gold = json.load(open(_GOLD, encoding="utf-8"))["items"]
    hard = [it for it in gold if it.get("hard")]
    llm = build_llm(s.llm, role="generation")

    pairs = run_gold_judge(
        hard, lambda a, c: judge_faithfulness(llm, a, c, n_samples=1, temperature=0.0))
    agr = judge_agreement(pairs)
    v = holistic_verdict(agr, mae_bar=mae_bar, residual_bar=residual_bar)

    def _f(x):
        return "—" if x is None else f"{x:.3f}"

    lines = [
        "# Подтверждение: холистический судья на трудных случаях",
        "",
        f"Трудных случаев: {len(hard)} (порог mae={mae_bar}, residual={residual_bar}).",
        "",
        f"- holistic judge: mae={_f(agr['mae'])}, направленный остаток={_f(agr['directional_residual'])}, "
        f"bucket-согласие={_f(agr['bucket_agreement'])} (n_scored={agr['n_scored']}, "
        f"воздержаний={agr['n_abstained']})",
        "",
        f"**Вердикт: {v['verdict']}** — "
        + {"artifact": "судья согласен с метками → 0.33 был артефактом; decompose-verify НЕ нужен.",
           "real_failure_bias": "судья систематически ошибается → рубрика/decompose-verify оправданы.",
           "real_failure_noise": "симметричный шум → аггрегация могла бы помочь.",
           "degenerate": "нет оценённых (всё воздержалось/пусто) — вердикт неопределён."}[v["verdict"]],
        "",
        "⚠️ n мал, how-to-архетип, метки посеяны агентом — вывод направленный, не статзначимый.",
    ]
    open(_CONFIRM_REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"DONE confirm hard: mae={_f(agr['mae'])} residual={_f(agr['directional_residual'])} "
          f"verdict={v['verdict']} -> {_CONFIRM_REPORT}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        main_confirm()
    else:
        main()
