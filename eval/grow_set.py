"""Рост набора оценки — главный рычаг доверия при малом n.

Расширяет существующие вопросы новыми (дедуп по нормализованному тексту), не заменяя.
Чистая `merge_dedup` тестируема; онлайн-раннер генерит вопросы через `question_gen`.

Целевой размер — ориентир под различимый эффект (≥ ~0.15): при малом paired-SD хватает
десятков вопросов, при большом — больше; точное n берётся из наблюдённого разброса дельт
после первого прогона (см. Open Questions плана). Раннер принимает `limit` аргументом.
"""

from __future__ import annotations

from eval.question_gen import _normalize


def merge_dedup(existing: list[dict], generated: list[dict]) -> list[dict]:
    """Существующие + новые вопросы без повторов по нормализованному тексту.

    Существующие сохраняются в приоритете и первыми; дубли (в т.ч. внутри `generated`)
    отбрасываются.
    """
    seen = {_normalize(q["question"]) for q in existing}
    out = list(existing)
    for q in generated:
        key = _normalize(q["question"])
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


T = "eval/trial"


def main() -> None:
    """Онлайн-раннер (нужны Neo4j + LLM): генерит `limit` вопросов из графа, сливает с
    текущими, пишет расширенный набор. Сама генерация — онлайн, вне оффлайн-тестов."""
    import json
    import sys

    from graphrag.config import load_settings
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm

    from eval.question_gen import generate_from_graph

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    existing = json.load(open(f"{T}/questions_real.json", encoding="utf-8"))

    s = load_settings()
    with Neo4jConnection(s.neo4j) as conn:
        llm = build_llm(s.llm, role="generation")
        generated = generate_from_graph(conn, llm, limit=limit)

    merged = merge_dedup(existing, generated)
    out_path = f"{T}/questions_grown.json"
    json.dump(merged, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"existing {len(existing)} + generated {len(generated)} -> {len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
