"""Парный A/B: lexical vs cross-encoder на одном наборе вопросов.

По каждому вопросу прогоняем обе версии, считаем по-вопросные дельты (только на
совместно-отвечённых, чтобы гасить шум сложности вопросов) и пер-версийные агрегаты
по собственному множеству отвечённых каждой версии (чтобы конверсия «воздержание →
ответ» была видна). Вердикт — по предрегистрации (строго положительная дельта на
min выборке; воздержания CE не выше lexical).
"""

from __future__ import annotations

from eval.quality_eval import evaluate_question


def is_answered(record: dict) -> bool:
    """Ответил ли (не воздержался): abstained.faithfulness True => воздержание."""
    return not record.get("abstained", {}).get("faithfulness", False)


def paired_deltas(pairs: list[dict], metric: str) -> list[dict]:
    """По-вопросные дельты `cross_encoder - lexical` на совместно-отвечённых вопросах.

    Вопрос учитывается, только если обе версии ответили И у обеих метрика не None.
    """
    out: list[dict] = []
    for p in pairs:
        a, b = p["lexical"], p["cross_encoder"]
        if is_answered(a) and is_answered(b):
            va = a["metrics"].get(metric)
            vb = b["metrics"].get(metric)
            if va is not None and vb is not None:
                out.append({
                    "question": p["question"],
                    "lexical": va,
                    "cross_encoder": vb,
                    "delta": vb - va,
                })
    return out


def per_version_summary(pairs: list[dict], version: str, metric: str) -> dict:
    """Среднее метрики по собственному множеству отвечённых данной версией вопросов."""
    vals = [
        p[version]["metrics"].get(metric)
        for p in pairs
        if is_answered(p[version]) and p[version]["metrics"].get(metric) is not None
    ]
    return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None}


def version_abstention(pairs: list[dict], version: str) -> dict:
    """Доля воздержаний версии по всему набору."""
    total = len(pairs)
    ab = sum(1 for p in pairs if not is_answered(p[version]))
    return {"total": total, "abstained": ab, "rate": ab / total if total else 0.0}


def permutation_pvalue(
    values: list[float], *, max_exact: int = 16, samples: int = 10000, seed: int = 0
) -> float:
    """Двусторонний p парного sign-flip перестановочного теста над средним.

    Под нулём знак каждой дельты случаен. При n <= max_exact — точный перебор всех
    `2^n` знаковых разворотов; иначе — Монте-Карло `samples` разворотов (numpy, сид).
    Точен при любом n, не опирается на хвосты бутстрапа.
    """
    import numpy as np

    v = np.asarray(list(values), dtype=float)
    n = len(v)
    if n == 0:
        return 1.0
    observed = abs(v.mean())
    if n <= max_exact:
        bits = (np.arange(2 ** n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
        signs = (1 - 2 * bits).astype(np.int8)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, n))
    means = np.abs((signs * v).mean(axis=1))
    return float((means >= observed - 1e-12).mean())


def n_needed(effect: float, sd: float, *, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Грубая двусторонняя оценка размера выборки для эффекта (ориентир, не гарантия)."""
    import math

    if effect == 0 or sd == 0:
        return None
    z = 1.96 + 0.84  # z_{alpha/2} (alpha=0.05) + z_{beta} (мощность 0.8)
    return int(math.ceil((z * sd / abs(effect)) ** 2))


def triage_verdict(deltas: list[dict], *, alpha: float = 0.05) -> dict:
    """Трёхисходный вердикт из по-вопросных дельт (соглашение: `delta = B − A`).

    Возвращает `{p, effect, verdict, n_needed}`: значимо и effect>0 -> `B_better`;
    значимо и effect<0 -> `A_better`; иначе -> `inconclusive` с оценкой `n_needed`.
    Помечается «на N курируемых вопросах» на уровне отчёта, не обобщается.
    """
    import numpy as np

    vals = [d["delta"] for d in deltas]
    if not vals:
        return {"p": 1.0, "effect": 0.0, "verdict": "inconclusive", "n_needed": None}
    arr = np.asarray(vals, dtype=float)
    effect = float(arr.mean())
    p = permutation_pvalue(vals)
    if p < alpha and effect != 0:
        return {"p": p, "effect": effect,
                "verdict": "B_better" if effect > 0 else "A_better", "n_needed": None}
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"p": p, "effect": effect, "verdict": "inconclusive",
            "n_needed": n_needed(effect, sd)}


_VERDICT_RU = {
    "A_better": "A (lexical) лучше",
    "B_better": "B (cross-encoder) лучше",
    "inconclusive": "неразрешимо",
}


def render_ab_report(pairs: list[dict], metrics: list[str]) -> str:
    """Markdown-отчёт A/B: пер-версийные средние, дельты, воздержания, триаж-вердикт."""
    lex_ab = version_abstention(pairs, "lexical")
    ce_ab = version_abstention(pairs, "cross_encoder")

    lines = [
        "# A/B: lexical vs cross-encoder",
        "",
        "⚠️ Само-оценка: одна модель и отвечает, и судит. Дельты индикативны.",
        "",
        f"Воздержания: lexical {lex_ab['abstained']}/{lex_ab['total']} "
        f"({lex_ab['rate']:.0%}) · cross-encoder {ce_ab['abstained']}/{ce_ab['total']} "
        f"({ce_ab['rate']:.0%})",
        "",
    ]
    for m in metrics:
        deltas = paired_deltas(pairs, m)
        lex = per_version_summary(pairs, "lexical", m)
        ce = per_version_summary(pairs, "cross_encoder", m)
        tv = triage_verdict(deltas)
        nn = f", нужно ~{tv['n_needed']} вопросов" if tv["n_needed"] else ""
        lines += [
            f"## {m}",
            "",
            f"- lexical: n={lex['n']} mean={_fmt(lex['mean'])} · "
            f"cross-encoder: n={ce['n']} mean={_fmt(ce['mean'])}",
            f"- совместно-отвечённых: {len(deltas)} · эффект {tv['effect']:+.3f}, "
            f"p={tv['p']:.3f} · вердикт: **{_VERDICT_RU[tv['verdict']]}**{nn} "
            "(на этих N вопросах, не обобщается)",
            "",
        ]
        if deltas:
            lines += ["| вопрос | lexical | cross-encoder | дельта |",
                      "|---|---|---|---|"]
            for d in deltas:
                q = d["question"][:60]
                lines.append(
                    f"| {q} | {_fmt(d['lexical'])} | {_fmt(d['cross_encoder'])} | "
                    f"{d['delta']:+.2f} |"
                )
            lines.append("")
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def changed_ranking(pairs: list[dict]) -> list[dict]:
    """Пары, где top-k (`context_ids`) различается между версиями."""
    return [
        p for p in pairs
        if p["lexical"].get("context_ids") != p["cross_encoder"].get("context_ids")
    ]


def render_changed_ranking(pairs: list[dict]) -> str:
    """Лист ручной судья-независимой сверки (R8) для изменившегося ранжирования.

    Автометрики нет — выход это человекочитаемый лист: по каждому изменившемуся
    вопросу оба контекста и оба ответа бок о бок для глазной адъюдикации.
    """
    changed = changed_ranking(pairs)
    lines = [
        "# Судья-независимая сверка: изменившееся ранжирование",
        "",
        "⚠️ Парный A/B НЕ контролирует смещение судьи под лечение (cross-encoder может "
        "поднимать фрагменты, которые судья любит независимо от истинной релевантности). "
        "Ниже — вопросы, где ранжирование изменилось; оцените релевантность контекста глазами.",
        "",
        "**Чек-лист аудита формы:** для каждого вопроса проверьте — не коррелирует ли "
        "предпочтение судьи с длиной/форматом ответа, а не с его релевантностью? Отметьте "
        "подозрительные (там смещение не симметрично, сравнение под вопросом).",
        "",
    ]
    if not changed:
        lines.append("Изменений ранжирования нет.")
        return "\n".join(lines)
    for p in changed:
        lex, ce = p["lexical"], p["cross_encoder"]
        lines += [
            f"## {p['question']}",
            "",
            "**lexical**",
            f"- context_ids: {lex.get('context_ids')}",
            f"- длина ответа: {len(lex.get('answer', ''))}",
            f"- ответ: {lex.get('answer', '')}",
            "",
            "**cross-encoder**",
            f"- context_ids: {ce.get('context_ids')}",
            f"- длина ответа: {len(ce.get('answer', ''))}",
            f"- ответ: {ce.get('answer', '')}",
            "",
        ]
    return "\n".join(lines)


def run_ab_eval(
    retr_lex,
    retr_ce,
    llm,
    questions: list[dict],
    labeled: list[dict] | None = None,
    *,
    evaluate_fn=evaluate_question,
) -> list[dict]:
    """Прогоняет каждый вопрос через обе версии; возвращает список пар записей.

    `evaluate_fn` инъектируется для тестов; по умолчанию — реальный `evaluate_question`.
    """
    items = [(q, None) for q in questions] + [
        (item, item.get("reference")) for item in (labeled or [])
    ]
    pairs: list[dict] = []
    for item, ref in items:
        # Устойчивость: сбой LLM на одном вопросе (таймаут и т.п.) не роняет весь
        # дорогой прогон — вопрос пропускается с предупреждением, остальные считаются.
        try:
            rec_lex = evaluate_fn(
                retr_lex, llm, item["question"], reference=ref, source_id=item.get("source_id")
            )
            rec_ce = evaluate_fn(
                retr_ce, llm, item["question"], reference=ref, source_id=item.get("source_id")
            )
        except Exception as e:  # noqa: BLE001 — намеренно широко: любой сбой = пропуск вопроса
            print(f"[skip] {item['question'][:50]}...: {e}", flush=True)
            continue
        pairs.append({
            "question": item["question"],
            "lexical": rec_lex,
            "cross_encoder": rec_ce,
        })
    return pairs


AB_METRICS = ("answer_relevance", "context_precision", "faithfulness")

T = "eval/trial"


def main() -> None:
    """Онлайн-прогон A/B на 13 реальных вопросах (нужны --extra ml + Neo4j + LLM).

    Вне оффлайн-тестового гейта: строит форсированный lexical и shipped-default
    cross-encoder, гоняет обе версии, рендерит отчёт и дампит парные записи.
    """
    import json

    import torch

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder
    from graphrag.embeddings.reranker import CrossEncoderReranker, LexicalReranker
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm

    from graphrag.retrieval.hybrid import HybridRetriever

    torch.set_num_threads(2)
    s = load_settings()
    questions = json.load(open(f"{T}/questions_real.json", encoding="utf-8"))
    labeled = json.load(open(f"{T}/labeled_real.json", encoding="utf-8"))

    with Neo4jConnection(s.neo4j) as conn:
        emb = build_embedder(s.embeddings)
        common = dict(top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
                      max_hops=s.retrieval.max_hops)
        retr_lex = HybridRetriever(conn, emb, LexicalReranker(), **common)
        retr_ce = HybridRetriever(conn, emb, CrossEncoderReranker(s.reranker.model), **common)
        llm = build_llm(s.llm, role="generation")

        pairs = run_ab_eval(retr_lex, retr_ce, llm, questions, labeled)

    report = render_ab_report(list(pairs), list(AB_METRICS))
    open(f"{T}/ab_report.md", "w", encoding="utf-8").write(report)
    open(f"{T}/ab_changed_ranking.md", "w", encoding="utf-8").write(
        render_changed_ranking(list(pairs))
    )
    json.dump({"pairs": pairs},
              open(f"{T}/ab_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # NB: не print(report) — отчёт содержит не-ASCII (эмодзи/кириллица), а Windows-
    # консоль в cp1251 падает на них. Отчёт уже записан в utf-8-файл.
    print(f"DONE {len(pairs)} pairs -> eval/trial/ab_report.md, ab_changed_ranking.md, ab_results.json")


if __name__ == "__main__":
    main()
