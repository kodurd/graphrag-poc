"""Пред-замер для verify-then-answer: сплит A/B на полюсе «0».

Вопрос: у скольких «неверных» ответов (faithfulness ≈0) после механического отсечения
неподкреплённых утверждений остаётся **обоснованное ядро** (тип A — «ластик» оставит
ответ), а сколько схлопнутся в пустоту (тип B — станут воздержанием, продуктовая
проблема покрытия корпуса, не инженерная).

Механизм: для каждой записи полюса «0» реконструируем контекст пере-извлечением, затем
LLM раскладывает зафиксированный ответ на атомарные утверждения и помечает каждое как
вытекающее из контекста или нет. Тип A = ≥1 подтверждённое утверждение.

Запуск (нужны --extra ml + Neo4j + LLM-ключ):
    uv run --extra ml python -m eval.claim_survival
"""

from __future__ import annotations

import json
import re

_CLAIM_PROMPT = (
    "Дан ОТВЕТ и КОНТЕКСТ. Разложи ОТВЕТ на атомарные фактические утверждения. Для каждого "
    "укажи, следует ли оно ПРЯМО из контекста (supported=true) или это домысел/обобщение, "
    "которого в контексте нет (supported=false). Оговорки, вопросы и фразы «в источнике не "
    "указано» — не утверждения, их пропусти. Верни ТОЛЬКО JSON: "
    '{"claims": [{"claim": "...", "supported": true|false}]}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_claims(raw: str) -> dict | None:
    """Парсит ответ судьи-декомпозера: {n_total, n_supported} либо None при мусоре."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    claims = data.get("claims")
    if not isinstance(claims, list):
        return None
    n_total = len(claims)
    n_supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
    return {"n_total": n_total, "n_supported": n_supported}


def classify(n_supported: int) -> str:
    """Тип A — есть обоснованное ядро (ластик оставит ответ); тип B — пустота → воздержание."""
    return "A" if n_supported >= 1 else "B"


_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_REPORT = "eval/trial/claim_survival_report.md"
_RESULTS = "eval/trial/claim_survival_results.json"


def main() -> int:
    from pathlib import Path

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.generate.answer import build_context
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.judge_noise import select_groups

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        print("claim-survival: не задан LLM_API_KEY (.env).")
        return 1

    records = json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]
    low = select_groups(records)["low"]
    print(f"claim-survival: полюс «0» = {len(low)} записей · reranker {s.reranker.provider}", flush=True)

    out: list[dict] = []
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("claim-survival: Neo4j недоступен — `docker compose up -d`")
            return 1
        retr = HybridRetriever(
            conn, build_embedder(s.embeddings), build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        llm = build_llm(s.llm, role="generation")
        for i, r in enumerate(low, 1):
            try:
                ctx = build_context(retr.retrieve(r["question"]).get("candidates", []))
                ctx_texts = "\n\n".join(c.text for c in ctx)
                prompt = f"{_CLAIM_PROMPT}\n\nКОНТЕКСТ:\n{ctx_texts}\n\nОТВЕТ:\n{r['answer']}"
                parsed = parse_claims(llm.complete(prompt))
                if parsed is None:
                    print(f"  [{i}/{len(low)}] parse-fail, пропуск", flush=True)
                    continue
                parsed.update({"type": classify(parsed["n_supported"]),
                               "route": r.get("route"), "source_id": r.get("source_id")})
                out.append(parsed)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(low)}] SKIP: {type(e).__name__}: {e}", flush=True)
                continue
            if i % 5 == 0:
                print(f"  [{i}/{len(low)}] обработано", flush=True)

    n = len(out)
    type_a = sum(1 for o in out if o["type"] == "A")
    type_b = n - type_a
    supported_counts = [o["n_supported"] for o in out]
    Path(_RESULTS).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    frac_a = type_a / n if n else 0.0
    med = sorted(supported_counts)[len(supported_counts) // 2] if supported_counts else 0
    lines = [
        "# Пред-замер verify-then-answer: сплит A/B на полюсе «0»",
        "",
        f"Записей оценено: {n} · тип A (есть обоснованное ядро): **{type_a}** "
        f"({frac_a:.0%}) · тип B (схлопнется в воздержание): **{type_b}** ({1-frac_a:.0%}).",
        f"Медиана подтверждённых утверждений на ответ: {med}.",
        "",
        "## Как читать",
        "- **Тип A** — «ластик» оставит обоснованный ответ: реальный инженерный выигрыш "
        "(faithfulness вверх, answer-rate держится). Это оправдывает постройку verify-then-answer.",
        "- **Тип B** — ядра нет, после стирания пустота → честное воздержание. Это продуктовая "
        "проблема (корпус не отвечает на вопрос), кодом не чинится.",
        "",
        f"**Вывод:** {'A преобладает → строить verify-then-answer оправдано' if frac_a >= 0.5 else 'B преобладает → рычаг продуктовый (корпус/скоуп), не пайплайн'} "
        f"(порог 50%).",
        "",
        "⚠️ Декомпозиция утверждений — та же LLM, что генерит и судит (само-оценка); оценка "
        "направленная, не абсолютная. Оценивает потенциал «ластика», не его реализацию.",
    ]
    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE claim-survival: A={type_a} B={type_b} ({frac_a:.0%} A) -> {_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
