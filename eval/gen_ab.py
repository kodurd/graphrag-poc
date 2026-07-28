"""A/B генератора: baseline vs claim-conservative (ANSWER_SYSTEM_STRICT).

Мотивация: снимок качества n=96 показал, что главный полюс низкой faithfulness — не
мусорный ретрив, а генератор, переобобщающий на ХОРОШЕМ контексте (23/28 провалов при
context_precision >= 0.6), сконцентрированный на маршрутах `mixed`/`multihop`. Этот
харнесс честно проверяет рычаг: на РЕАЛЬНОМ прод-контексте (cross-encoder) обе руки —
baseline и strict — генерят на ОДНОМ И ТОМ ЖЕ извлечённом контексте, судья валидированным
холистическим судьёй оценивает faithfulness и воздержания. Выборка сужена до overreach-
склонных маршрутов (route in {mixed, multihop}) прямо из ретривера.

Решение о шипе — предрегистрированный критерий (`verdict`): faithfulness растёт значимо
(парный перестановочный тест) И воздержания не растут больше guardrail. Это защищает от
повторения истории fusion, который «улучшал» ценой роста воздержаний.

Запуск (нужны --extra ml + Neo4j с корпусом + LLM-ключ):
    uv run --extra ml python -m eval.gen_ab
"""

from __future__ import annotations

# Порог значимости и guardrail воздержаний — предрегистрированы (не подгоняются под итог).
P_THRESHOLD = 0.05
ABSTENTION_GUARDRAIL = 0.05  # +5 п.п.; fusion зарубили на ~+10 п.п.
_OVERREACH_ROUTES = ("mixed", "multihop")


def mean_faith(scores: list[float | None]) -> float | None:
    """Средняя faithfulness по оценённым (None — воздержание/сбой судьи — вне среднего)."""
    vals = [s for s in scores if s is not None]
    return sum(vals) / len(vals) if vals else None


def compare_arms(
    baseline_scores: list[float | None], strict_scores: list[float | None]
) -> dict:
    """Средние faithfulness по плечам + дельта средних (strict − baseline).

    Дельта средних — грубый ориентир; строгий вердикт считает `verdict` по ПАРНЫМ дельтам.
    Дельта None, если у какого-то плеча нет оценённых ответов.
    """
    b = mean_faith(baseline_scores)
    s = mean_faith(strict_scores)
    delta = (s - b) if (b is not None and s is not None) else None
    return {"baseline_mean": b, "strict_mean": s, "delta": delta,
            "n_baseline": sum(1 for x in baseline_scores if x is not None),
            "n_strict": sum(1 for x in strict_scores if x is not None)}


def verdict(
    pairs: list[dict],
    *,
    p_threshold: float = P_THRESHOLD,
    abstention_guardrail: float = ABSTENTION_GUARDRAIL,
) -> dict:
    """Предрегистрированный критерий ship/no-ship по парному A/B.

    `pairs` — по-вопросные записи с ключами base_score, base_abstained, strict_score,
    strict_abstained (score: float|None; abstained: bool). Ship, если:
      1) faithfulness растёт значимо: парный перестановочный тест p <= порог И
         средняя парная дельта > 0 (на совместно-оценённых вопросах), И
      2) воздержания не растут больше guardrail: Δabstention <= порог.
    Оба гейта обязательны — рост faithfulness ценой воздержаний не проходит (урок fusion).
    """
    from eval.ab_eval import permutation_pvalue

    n_total = len(pairs)
    # Парные дельты только там, где ОБЕ руки дали число (0.0 — валидный балл, не «нет данных»).
    deltas = [
        p["strict_score"] - p["base_score"]
        for p in pairs
        if p.get("base_score") is not None and p.get("strict_score") is not None
    ]
    base_abst = sum(1 for p in pairs if p.get("base_abstained"))
    strict_abst = sum(1 for p in pairs if p.get("strict_abstained"))
    base_abst_rate = base_abst / n_total if n_total else 0.0
    strict_abst_rate = strict_abst / n_total if n_total else 0.0
    abstention_delta = strict_abst_rate - base_abst_rate

    if not deltas:
        return {
            "n_total": n_total, "n_paired": 0, "faith_delta": None, "p": 1.0,
            "base_abst_rate": base_abst_rate, "strict_abst_rate": strict_abst_rate,
            "abstention_delta": abstention_delta, "ship": False,
            "reason": "нет совместно-оценённых пар — вердикт невозможен",
        }

    faith_delta = sum(deltas) / len(deltas)
    p = permutation_pvalue(deltas)
    significant = faith_delta > 0 and p <= p_threshold
    guardrail_ok = abstention_delta <= abstention_guardrail
    ship = significant and guardrail_ok

    if ship:
        reason = "strict значимо вернее без роста воздержаний выше guardrail"
    elif not significant:
        reason = f"нет значимого роста faithfulness (дельта {faith_delta:+.3f}, p={p:.3f})"
    else:
        reason = (f"faithfulness растёт (дельта {faith_delta:+.3f}, p={p:.3f}), но "
                  f"воздержания +{abstention_delta:.1%} > guardrail {abstention_guardrail:.0%}")

    return {
        "n_total": n_total, "n_paired": len(deltas), "faith_delta": faith_delta, "p": p,
        "base_abst_rate": base_abst_rate, "strict_abst_rate": strict_abst_rate,
        "abstention_delta": abstention_delta, "ship": ship, "reason": reason,
    }


_REPORT = "eval/trial/gen_ab_real_report.md"
_RESULTS = "eval/trial/gen_ab_real_results.json"
_QUESTIONS = "eval/trial/questions_grown.json"


def main_real(limit: int | None = None) -> None:
    """Paired A/B на реальном retrieved-контексте, суженный до overreach-маршрутов.

    Для каждого вопроса — реальный retrieval (cross-encoder, прод-конвейер); если маршрут
    не в {mixed, multihop} — пропуск (переобобщение там не концентрируется). Иначе обе руки
    (baseline vs strict) генерят на ОДНОМ контексте, судья считает faithfulness + воздержание.
    Нужны --extra ml + Neo4j + LLM. `limit` — подвыборка вопросов до фильтра (для экономии).
    """
    import json

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.generate.answer import (
        ANSWER_SYSTEM,
        ANSWER_SYSTEM_STRICT,
        build_context,
        generate_answer,
    )
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.metrics import judge_faithfulness

    s = load_settings()
    questions = json.load(open(_QUESTIONS, encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]

    pairs: list[dict] = []
    skipped_route = 0

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("gen-ab: Neo4j недоступен — `docker compose up -d`")
            return
        emb = build_embedder(s.embeddings)
        retr = HybridRetriever(
            conn, emb, build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        llm = build_llm(s.llm, role="generation")
        print(
            f"gen-ab: вопросов {len(questions)} · reranker {s.reranker.provider} · "
            f"фильтр route in {_OVERREACH_ROUTES}",
            flush=True,
        )
        for i, item in enumerate(questions, 1):
            q = item["question"]
            try:
                retrieved = retr.retrieve(q)
                route = retrieved.get("route")
                if route not in _OVERREACH_ROUTES:
                    skipped_route += 1
                    continue
                ctx = build_context(retrieved.get("candidates", []))
                ctx_texts = [c.text for c in ctx]
                rec: dict = {"question": q, "route": route,
                             "source_id": item.get("source_id")}
                for system, prefix in ((ANSWER_SYSTEM, "base"), (ANSWER_SYSTEM_STRICT, "strict")):
                    ans = generate_answer(llm, q, ctx, system=system)
                    score, abstained = judge_faithfulness(llm, ans.text, ctx_texts)
                    rec[f"{prefix}_score"] = score
                    rec[f"{prefix}_abstained"] = abstained
                pairs.append(rec)
            except Exception as e:  # noqa: BLE001 — сбой на вопросе не роняет прогон
                print(f"  [{i}/{len(questions)}] SKIP: {type(e).__name__}: {e}", flush=True)
                continue
            if len(pairs) % 5 == 0:
                print(f"  [{i}/{len(questions)}] overreach-пар: {len(pairs)}", flush=True)

    v = verdict(pairs)
    base_scores = [p.get("base_score") for p in pairs]
    strict_scores = [p.get("strict_score") for p in pairs]
    cmp = compare_arms(base_scores, strict_scores)

    json.dump({"pairs": pairs, "verdict": v, "compare": cmp},
              open(_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def _f(x):
        return "—" if x is None else f"{x:.3f}"

    lines = [
        "# A/B генератора: baseline vs claim-conservative (реальный контекст, overreach-маршруты)",
        "",
        f"Вопросов рассмотрено: {len(questions)} · пропущено по маршруту "
        f"(не mixed/multihop): {skipped_route} · overreach-пар: {v['n_total']}.",
        "Реальный retrieval (cross-encoder), обе руки на общем контексте, judged холистическим судьёй.",
        "",
        "## Плечи (среднее faithfulness)",
        f"- baseline: mean={_f(cmp['baseline_mean'])} (оценено {cmp['n_baseline']}, "
        f"воздержаний {sum(1 for p in pairs if p.get('base_abstained'))})",
        f"- claim-conservative: mean={_f(cmp['strict_mean'])} (оценено {cmp['n_strict']}, "
        f"воздержаний {sum(1 for p in pairs if p.get('strict_abstained'))})",
        "",
        "## Вердикт (предрегистрированный критерий)",
        f"- совместно-оценённых пар: {v['n_paired']}",
        f"- парная дельта faithfulness (strict − baseline): {_f(v['faith_delta'])} · "
        f"p={v['p']:.3f} (порог {P_THRESHOLD})",
        f"- воздержания: baseline {v['base_abst_rate']:.1%} → strict {v['strict_abst_rate']:.1%} "
        f"(Δ {v['abstention_delta']:+.1%}, guardrail +{ABSTENTION_GUARDRAIL:.0%})",
        "",
        f"**Решение: {'SHIP' if v['ship'] else 'NO-SHIP'}** — {v['reason']}",
        "",
        "⚠️ Само-судья, судья невоспроизводим между прогонами; paired-дизайн на общем контексте "
        "снижает часть шума. n на overreach-подмножестве мал — при неразрешимости тест это покажет.",
    ]
    open(_REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print(
        f"DONE gen A/B: baseline={_f(cmp['baseline_mean'])} strict={_f(cmp['strict_mean'])} "
        f"paired_delta={_f(v['faith_delta'])} p={v['p']:.3f} "
        f"abst_delta={v['abstention_delta']:+.1%} -> {'SHIP' if v['ship'] else 'NO-SHIP'}",
        flush=True,
    )


if __name__ == "__main__":
    main_real()
