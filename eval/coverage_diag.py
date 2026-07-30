"""Диагностика покрытия how-to: отсутствует/не-поднят vs не извлекается прод-пулом.

Measure-first шаг к верному how-to-ответчику. Для проваленных how-to-вопросов на двух уровнях
пула (прод cross-encoder vs широкий высокий-top_k) считаем двойной сигнал: LLM-судья «хватает ли
пула ответить без домысла» И объективный `source_id`-якорь (попал ли первоисточник в пул, как в
`recall_gate.py`). Плюс gold-probe пола. Классификация → вердикт направления с зоной
неопределённости, гейтящий дальнейшую работу по покрытию (ingest vs ретрив — не в этом модуле).

Ядро (этот модуль, U1) — чистые функции без сети. Раннер (`main`, U2) собирает пулы, судит и
пишет отчёт.

Запуск (нужны --extra ml + Neo4j + LLM-ключ):
    uv run --extra ml python -m eval.coverage_diag
"""

from __future__ import annotations

import json
import re

# Предрегистрированные пороги вердикта (зона неопределённости против near-tie).
NOT_SURFACED_HIGH = 0.60  # доля not_surfaced ≥ → «источники»
NOT_SURFACED_LOW = 0.40   # доля not_surfaced ≤ → «ретрив»; между — ambiguous
_OVERREACH_ROUTES = ("mixed", "multihop")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def select_failed_howto(records: list[dict]) -> list[dict]:
    """Проваленные how-to: маршруты mixed/multihop, где воздержание ИЛИ faithfulness < 0.2.

    None-guard обязателен: у воздержавшихся faithfulness = null; наивный `None < 0.2` роняет
    отбор. Порядок: сперва флаг воздержания, порог — только при не-None.
    """
    out: list[dict] = []
    for r in records:
        if r.get("route") not in _OVERREACH_ROUTES:
            continue
        abstained = bool((r.get("abstained") or {}).get("faithfulness"))
        f = (r.get("metrics") or {}).get("faithfulness")
        if abstained or (f is not None and float(f) < 0.2):
            out.append(r)
    return out


def parse_coverage(raw: str) -> dict | None:
    """Парс вердикта судьи покрытия → {"sufficient": bool} либо None при мусоре/сбое.

    None ≠ False: сбой судьи не приводить к «недостаточно» (дисциплина проекта — None ≠ 0).
    """
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if "sufficient" not in data or not isinstance(data["sufficient"], bool):
        return None
    return {"sufficient": data["sufficient"]}


def classify(prod_suff: bool | None, broad_suff: bool | None) -> str:
    """Класс вопроса по достаточности прод- и широкого пула.

    None на любом уровне (сбой судьи) → `judge_failed` (НЕ приводить к False).
    """
    if prod_suff is None or broad_suff is None:
        return "judge_failed"
    if prod_suff:
        return "ranked"           # хватает прод-пула
    if broad_suff:
        return "unretrieved"      # хватает широкого, не прод — ретрив/чанкинг
    return "not_surfaced"         # не поднят даже широко — отсутствие ИЛИ глубокий провал ретрива


def aggregate(classes: list[str]) -> dict:
    """Доли по не-ranked базе (unretrieved + not_surfaced); judge_failed вне знаменателя.

    Вердикт с зоной неопределённости: not_surfaced-доля ≥0.60 → «источники»; ≤0.40 → «ретрив»;
    иначе `ambiguous`.
    """
    from collections import Counter
    c = Counter(classes)
    base = c["unretrieved"] + c["not_surfaced"]
    not_surfaced_frac = c["not_surfaced"] / base if base else None

    if not_surfaced_frac is None:
        verdict, reason = "ambiguous", "нет оценённых пар в базе (unretrieved+not_surfaced)"
    elif not_surfaced_frac >= NOT_SURFACED_HIGH:
        verdict, reason = "sources", f"not_surfaced {not_surfaced_frac:.0%} ≥ {NOT_SURFACED_HIGH:.0%} — нужны новые источники"
    elif not_surfaced_frac <= NOT_SURFACED_LOW:
        verdict, reason = "retrieval", f"not_surfaced {not_surfaced_frac:.0%} ≤ {NOT_SURFACED_LOW:.0%} — ретрив/чанкинг"
    else:
        verdict, reason = "ambiguous", f"not_surfaced {not_surfaced_frac:.0%} в зоне неопределённости {NOT_SURFACED_LOW:.0%}–{NOT_SURFACED_HIGH:.0%} — увеличить выборку / не гейтить"

    return {
        "counts": dict(c),
        "n_total": len(classes),
        "base": base,                       # не-ranked, не judge_failed
        "judge_failed": c["judge_failed"],  # вне знаменателя
        "not_surfaced_frac": not_surfaced_frac,
        "verdict": verdict,
        "reason": reason,
    }


# --- Раннер (U2): два уровня пула, судья + source_id, gold-probe ---

BROAD_TOP_K = 50   # широкий пул: высокий top_k через _candidate_pool
JUDGE_BATCH = 8    # чанков на батч судьи (OR-агрегация по батчам)
GOLD_N = 8         # размер gold-probe (заведомо-покрытые вопросы)

_COVERAGE_PROMPT = (
    "Дан ВОПРОС и фрагменты КОНТЕКСТА. Хватает ли ЭТОГО контекста, чтобы ответить на вопрос "
    "ВЕРНО и БЕЗ ДОМЫСЛА (только то, что прямо следует из фрагментов)? "
    'Верни ТОЛЬКО JSON: {"sufficient": true|false}'
)

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_REPORT = "eval/trial/coverage_diag_report.md"
_RESULTS = "eval/trial/coverage_diag_results.json"


def coverage_judge(llm, question: str, texts: list[str], *, batch: int = JUDGE_BATCH) -> bool | None:
    """Хватает ли пула ответить без домысла — батчами по всему пулу с OR-агрегацией.

    Отбор чанков не подменяет тест наличия: судим ВЕСЬ пул порциями, «хватает», если хватает
    хоть в одном батче. None, если ни один батч не распарсился (сбой судьи ≠ «не хватает»).
    """
    if not texts:
        return False
    any_parsed = False
    for i in range(0, len(texts), batch):
        chunk = "\n\n".join(texts[i:i + batch])
        prompt = f"{_COVERAGE_PROMPT}\n\nВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{chunk}"
        try:
            parsed = parse_coverage(llm.complete(prompt))
        except Exception:  # noqa: BLE001 — сбой одного батча не роняет
            parsed = None
        if parsed is not None:
            any_parsed = True
            if parsed["sufficient"]:
                return True
    return False if any_parsed else None


def _judge_question(llm, retr_prod, retr_broad, rec: dict) -> dict:
    """Собрать оба пула, судить достаточность и source_id-recall по каждому, классифицировать."""
    from eval.recall_gate import source_in_pool

    q = rec["question"]
    sid = rec.get("source_id") or ""
    prod_cands = retr_prod.retrieve(q).get("candidates", [])
    route, broad_pool = retr_broad._candidate_pool(q)

    prod_texts = [c.get("text", "") for c in prod_cands]
    broad_texts = [c.get("text", "") for c in broad_pool]
    prod_suff = coverage_judge(llm, q, prod_texts)
    broad_suff = coverage_judge(llm, q, broad_texts)
    cls = classify(prod_suff, broad_suff)

    # source_id-якорь; на graph-only multihop-пуле source_id даёт 0 по построению — помечаем n/a.
    sid_applicable = route != "multihop"
    return {
        "question": q, "route": route, "source_id": sid,
        "prod_sufficient": prod_suff, "broad_sufficient": broad_suff, "class": cls,
        "sid_in_prod": source_in_pool(prod_cands, sid) if (sid and sid_applicable) else None,
        "sid_in_broad": source_in_pool(broad_pool, sid) if (sid and sid_applicable) else None,
        "sid_applicable": sid_applicable,
    }


def main() -> int:
    from pathlib import Path

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        print("coverage-diag: не задан LLM_API_KEY (.env).")
        return 1

    records = json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]
    failed = select_failed_howto(records)
    # gold-probe пола: заведомо-покрытые (высокая faithfulness) — метод обязан дать retrievable.
    gold = [r for r in records
            if (r.get("metrics") or {}).get("faithfulness") is not None
            and float(r["metrics"]["faithfulness"]) >= 0.8][:GOLD_N]
    print(f"coverage-diag: проваленных how-to {len(failed)} · gold-probe {len(gold)} · "
          f"широкий top_k={BROAD_TOP_K}", flush=True)

    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("coverage-diag: Neo4j недоступен — `docker compose up -d`")
            return 1
        emb = build_embedder(s.embeddings)
        retr_prod = HybridRetriever(
            conn, emb, build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        retr_broad = HybridRetriever(
            conn, emb, build_reranker(s.reranker),
            top_k=BROAD_TOP_K, rerank_top_k=BROAD_TOP_K,
            max_hops=s.retrieval.max_hops, min_rerank_score=0.0,
        )
        llm = build_llm(s.llm, role="generation")

        def run(items, tag):
            out = []
            for i, rec in enumerate(items, 1):
                try:
                    out.append(_judge_question(llm, retr_prod, retr_broad, rec))
                except Exception as e:  # noqa: BLE001
                    print(f"  [{tag} {i}/{len(items)}] SKIP: {type(e).__name__}: {e}", flush=True)
                if i % 5 == 0:
                    print(f"  [{tag}] {i}/{len(items)}", flush=True)
            return out

        failed_res = run(failed, "failed")
        gold_res = run(gold, "gold")

    v = aggregate([r["class"] for r in failed_res])
    # gold-probe: доля retrievable (не not_surfaced) среди gold — должна быть высокой.
    gold_retrievable = sum(1 for r in gold_res if r["class"] in ("ranked", "unretrieved"))
    gold_pass = gold_retrievable / len(gold_res) if gold_res else None
    trusted = gold_pass is not None and gold_pass >= 0.75
    # divergence: source_id в широком пуле, но судья говорит not_surfaced (разоблачает конфляцию).
    divergence = sum(1 for r in failed_res
                     if r["class"] == "not_surfaced" and (r.get("sid_in_broad") or 0) >= 1.0)

    Path(_RESULTS).write_text(json.dumps(
        {"failed": failed_res, "gold": gold_res, "verdict": v,
         "gold_pass": gold_pass, "trusted": trusted, "divergence": divergence},
        ensure_ascii=False, indent=2), encoding="utf-8")

    def _f(x):
        return "—" if x is None else (f"{x:.0%}" if isinstance(x, float) else str(x))

    lines = [
        "# Диагностика покрытия how-to",
        "",
        f"Проваленных how-to: {v['n_total']} · база (не-ranked): {v['base']} · "
        f"gold-probe: {len(gold_res)}. Двойной сигнал: судья покрытия + source_id-якорь.",
        "",
        "## Классы (проваленные how-to)",
        f"- {v['counts']}",
        f"- not_surfaced доля (по не-ranked базе): {_f(v['not_surfaced_frac'])} · "
        f"judge_failed (вне знаменателя): {v['judge_failed']}",
        f"- divergence (source_id в широком пуле, но not_surfaced): {divergence} — "
        "кандидаты «источник есть, ответа-текста в нём нет»",
        "",
        "## Проверка пола (gold-probe)",
        f"- заведомо-покрытые retrievable: {gold_retrievable}/{len(gold_res)} ({_f(gold_pass)})",
        f"- метод {'ДОВЕРЕННЫЙ' if trusted else 'НЕ ДОВЕРЕННЫЙ (gold-probe провалился)'} "
        "(порог 75%)",
        "",
        f"**Вердикт направления: {v['verdict'].upper()}** — {v['reason']}"
        + ("" if trusted else " ⚠️ вердикт НЕ доверенный: gold-probe не прошёл."),
        "",
        "⚠️ Судья — само-оценка, консервативен; not_surfaced не разделяет отсутствие от глубокой "
        "неизвлекаемости идеально (source_id-recall разводит частично). Выборка мала — вердикт "
        "грубый (направление, не точность).",
    ]
    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE coverage-diag: verdict={v['verdict']} not_surfaced={_f(v['not_surfaced_frac'])} "
          f"gold_pass={_f(gold_pass)} trusted={trusted} -> {_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
