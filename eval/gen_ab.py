"""A/B генератора: baseline-промпт vs claim-conservative (ANSWER_SYSTEM_STRICT).

Дёшево: фикстуры — трудные случаи из `faith_gold.json` (вопрос + инлайн-контекст), ни
Neo4j, ни ретрива. Перегенерируем ответ обоими промптами, судим faithfulness ВАЛИДИРОВАННЫМ
холистическим судьёй и сравниваем среднюю подкреплённость. Проверяем: убирает ли
claim-conservative переобобщение (растёт ли faithfulness), не роняя ответы в воздержание.
"""

from __future__ import annotations


def mean_faith(scores: list[float | None]) -> float | None:
    """Средняя faithfulness по оценённым (None — воздержание/сбой судьи — вне среднего)."""
    vals = [s for s in scores if s is not None]
    return sum(vals) / len(vals) if vals else None


def compare_arms(
    baseline_scores: list[float | None], strict_scores: list[float | None]
) -> dict:
    """Сравнение плеч: средние faithfulness + дельта (strict − baseline). Дельта None, если
    у какого-то плеча нет оценённых ответов."""
    b = mean_faith(baseline_scores)
    s = mean_faith(strict_scores)
    delta = (s - b) if (b is not None and s is not None) else None
    return {"baseline_mean": b, "strict_mean": s, "delta": delta,
            "n_baseline": sum(1 for x in baseline_scores if x is not None),
            "n_strict": sum(1 for x in strict_scores if x is not None)}


_GOLD = "eval/trial/faith_gold.json"
_REPORT = "eval/trial/gen_ab_report.md"


def main() -> None:
    """Онлайн: перегенерировать трудные вопросы baseline vs strict и сравнить faithfulness.

    Нужен LLM. Дёшево (только hard-подмножество, контекст инлайн). Пишет отчёт с дельтой.
    """
    import json

    from graphrag.config import load_settings
    from graphrag.generate.answer import ANSWER_SYSTEM, ANSWER_SYSTEM_STRICT, ContextItem, generate_answer
    from graphrag.llm import build_llm

    from eval.metrics import judge_faithfulness

    s = load_settings()
    gold = json.load(open(_GOLD, encoding="utf-8"))["items"]
    hard = [it for it in gold if it.get("hard")]
    llm = build_llm(s.llm, role="generation")

    def _faith_for(system: str) -> list[float | None]:
        scores: list[float | None] = []
        for it in hard:
            ctx = [ContextItem(text=it["context_text"], uri=f"gold://{it['id']}")]
            ans = generate_answer(llm, it["question"], ctx, system=system)
            score, _ = judge_faithfulness(llm, ans.text, [it["context_text"]], n_samples=1, temperature=0.0)
            scores.append(score)
        return scores

    baseline = _faith_for(ANSWER_SYSTEM)
    strict = _faith_for(ANSWER_SYSTEM_STRICT)
    cmp = compare_arms(baseline, strict)

    def _f(x):
        return "—" if x is None else f"{x:.3f}"

    verdict = ("claim-conservative ВЕРНЕЕ (дельта > 0) — убрать переобобщение помогает."
               if (cmp["delta"] or 0) > 0.02
               else "нет улучшения faithfulness — переобобщение не главный источник, либо нужен другой промпт.")
    lines = [
        "# A/B генератора: baseline vs claim-conservative",
        "",
        f"Трудных вопросов: {len(hard)} (перегенерация, judged холистическим судьёй, temp=0).",
        "",
        f"- baseline faithfulness: mean={_f(cmp['baseline_mean'])} (n={cmp['n_baseline']})",
        f"- claim-conservative: mean={_f(cmp['strict_mean'])} (n={cmp['n_strict']})",
        f"- дельта (strict − baseline): {_f(cmp['delta'])}",
        "",
        f"**Вывод:** {verdict}",
        "",
        "⚠️ n мал, how-to-архетип — вывод направленный, не статзначимый.",
    ]
    open(_REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"DONE gen A/B: baseline={_f(cmp['baseline_mean'])} strict={_f(cmp['strict_mean'])} "
          f"delta={_f(cmp['delta'])} -> {_REPORT}")


if __name__ == "__main__":
    main()
