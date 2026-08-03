"""A/B: baseline vs verify-then-answer на реальном retrieved-контексте.

Первичный несмешанный сигнал успеха — answer_relevance (не ниже baseline) + доля residual +
воздержания. Faithfulness измеряется ХОЛИСТИЧЕСКИМ судьёй (`judge_faithfulness`, промпт
отличается от фильтрующего декомпозера в verify.py) как ПОДТВЕРЖДАЮЩИЙ сигнал: рост faithfulness
у руки B частично тавтологичен (B собран из того, что фильтр счёл подтверждённым), поэтому не
может быть первичным гейтом. Выборка сужена до overreach-маршрутов (mixed/multihop).

Запуск (нужны --extra ml + Neo4j + LLM-ключ):
    uv run --extra ml python -m eval.verify_ab
"""

from __future__ import annotations

# Предрегистрированные пороги.
RELEVANCE_GUARDRAIL = 0.05  # answer_relevance не должен просесть больше чем на это
ABSTENTION_GUARDRAIL = 0.05  # +5 п.п. к воздержаниям — потолок
RESIDUAL_NORM = 0.2  # средняя доля остаточного домысла у руки B — потолок
P_THRESHOLD = 0.05
_OVERREACH_ROUTES = ("mixed", "multihop")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _paired_deltas(pairs: list[dict], a_key: str, b_key: str) -> list[float]:
    """Парные дельты b−a на вопросах, где обе руки ответили и метрика не None."""
    out: list[float] = []
    for p in pairs:
        if p.get("a_abst") or p.get("b_abst"):
            continue
        a, b = p.get(a_key), p.get(b_key)
        if a is not None and b is not None:
            out.append(b - a)
    return out


def verdict(
    pairs: list[dict],
    *,
    relevance_guardrail: float = RELEVANCE_GUARDRAIL,
    abstention_guardrail: float = ABSTENTION_GUARDRAIL,
    residual_norm: float = RESIDUAL_NORM,
    p_threshold: float = P_THRESHOLD,
) -> dict:
    """Ship по ПЕРВИЧНЫМ несмешанным гейтам: relevance не просел И воздержания ≤ guardrail И
    residual в норме. Faithfulness (другой судья) — подтверждающий, не блокирует сам по себе."""
    from eval.ab_eval import permutation_pvalue

    n = len(pairs)
    a_abst = sum(1 for p in pairs if p.get("a_abst")) / n if n else 0.0
    b_abst = sum(1 for p in pairs if p.get("b_abst")) / n if n else 0.0
    abstention_delta = b_abst - a_abst

    rel_deltas = _paired_deltas(pairs, "a_rel", "b_rel")
    rel_mean = _mean(rel_deltas)
    rel_p = permutation_pvalue(rel_deltas) if rel_deltas else 1.0

    residuals = [p["residual"] for p in pairs
                 if not p.get("b_abst") and p.get("residual") is not None]
    residual_mean = _mean(residuals)

    faith_deltas = _paired_deltas(pairs, "a_faith", "b_faith")  # подтверждающий
    faith_mean = _mean(faith_deltas)
    faith_p = permutation_pvalue(faith_deltas) if faith_deltas else 1.0

    # Первичные гейты (falsy-zero-безопасно через явные None-проверки).
    relevance_ok = rel_mean is not None and rel_mean >= -relevance_guardrail
    abstention_ok = abstention_delta <= abstention_guardrail
    residual_ok = residual_mean is None or residual_mean <= residual_norm
    ship = relevance_ok and abstention_ok and residual_ok

    if ship:
        reason = "первичные гейты пройдены (relevance держится, воздержания и residual в норме)"
    else:
        fails = []
        if not relevance_ok:
            rm = "—" if rel_mean is None else f"{rel_mean:+.3f}"
            fails.append(f"relevance просел ({rm})")
        if not abstention_ok:
            fails.append(f"воздержания +{abstention_delta:.1%} > {abstention_guardrail:.0%}")
        if not residual_ok:
            fails.append(f"residual {residual_mean:.2f} > {residual_norm:.2f}")
        reason = "no-ship: " + "; ".join(fails)

    return {
        "n": n, "ship": ship, "reason": reason,
        "relevance_delta": rel_mean, "relevance_p": rel_p,
        "abstention_delta": abstention_delta, "a_abst_rate": a_abst, "b_abst_rate": b_abst,
        "residual_mean": residual_mean,
        "faith_delta": faith_mean, "faith_p": faith_p,  # подтверждающий
    }


_REPORT = "eval/trial/verify_ab_report.md"
_RESULTS = "eval/trial/verify_ab_results.json"
_QUESTIONS = "eval/trial/questions_grown.json"


def main() -> int:
    import json
    from pathlib import Path

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.generate.answer import build_context, generate_answer
    from graphrag.generate.verify import verify_then_answer
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.metrics import judge_answer_relevance, judge_faithfulness

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        print("verify-ab: не задан LLM_API_KEY (.env).")
        return 1
    questions = json.loads(Path(_QUESTIONS).read_text(encoding="utf-8"))

    pairs: list[dict] = []
    skipped_route = 0
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("verify-ab: Neo4j недоступен — `docker compose up -d`")
            return 1
        retr = HybridRetriever(
            conn, build_embedder(s.embeddings), build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        llm = build_llm(s.llm, role="generation")
        print(f"verify-ab: вопросов {len(questions)} · фильтр route in {_OVERREACH_ROUTES}", flush=True)
        for i, item in enumerate(questions, 1):
            q = item["question"]
            try:
                retrieved = retr.retrieve(q)
                if retrieved.get("route") not in _OVERREACH_ROUTES:
                    skipped_route += 1
                    continue
                ctx = build_context(retrieved.get("candidates", []))
                ctx_texts = [c.text for c in ctx]
                a = generate_answer(llm, q, ctx)  # baseline
                b = verify_then_answer(llm, q, a.text, ctx)  # verify
                rec = {
                    "question": q,
                    "a_abst": not a.grounded and not a.text.strip(),  # baseline редко воздерживается
                    "b_abst": b.abstained,
                    "residual": b.residual_frac,
                    "a_rel": judge_answer_relevance(llm, q, a.text),
                    "b_rel": None if b.abstained else judge_answer_relevance(llm, q, b.text),
                    "a_faith": judge_faithfulness(llm, a.text, ctx_texts)[0],
                    "b_faith": None if b.abstained else judge_faithfulness(llm, b.text, ctx_texts)[0],
                }
                pairs.append(rec)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(questions)}] SKIP: {type(e).__name__}: {e}", flush=True)
                continue
            if len(pairs) % 5 == 0:
                print(f"  [{i}/{len(questions)}] пар: {len(pairs)}", flush=True)

    v = verdict(pairs)
    json.dump({"pairs": pairs, "verdict": v}, open(_RESULTS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    def _f(x):
        return "—" if x is None else f"{x:.3f}"

    lines = [
        "# A/B: baseline vs verify-then-answer (overreach-маршруты)",
        "",
        f"Пар: {v['n']} · пропущено по маршруту: {skipped_route}. Первичный сигнал — relevance + "
        "residual + воздержания; faithfulness — подтверждающий (холистический судья, промпт ≠ фильтр).",
        "",
        "## Первичные гейты",
        f"- answer_relevance дельта (B−A): {_f(v['relevance_delta'])} (p={v['relevance_p']:.3f}; "
        f"guardrail −{RELEVANCE_GUARDRAIL:.0%})",
        f"- воздержания: A {v['a_abst_rate']:.1%} → B {v['b_abst_rate']:.1%} "
        f"(Δ {v['abstention_delta']:+.1%}; guardrail +{ABSTENTION_GUARDRAIL:.0%})",
        f"- средняя доля residual у B: {_f(v['residual_mean'])} (норма ≤ {RESIDUAL_NORM:.2f})",
        "",
        "## Подтверждающий (не блокирует)",
        f"- faithfulness дельта (B−A): {_f(v['faith_delta'])} (p={v['faith_p']:.3f}) — другой судья",
        "",
        f"**Решение: {'SHIP' if v['ship'] else 'NO-SHIP'}** — {v['reason']}",
        "",
        "⚠️ Само-оценка; faithfulness частично циркулярен (потому и подтверждающий, не первичный).",
    ]
    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE verify-ab: rel={_f(v['relevance_delta'])} abst={v['abstention_delta']:+.1%} "
          f"residual={_f(v['residual_mean'])} -> {'SHIP' if v['ship'] else 'NO-SHIP'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
