"""Замер связи тикет→фикс-коммит: surfacing vs ingest vs content_problem.

Measure-first к «поднимать связанный коммит-фикс в how-to-контекст». Для проваленных
how-to-тикетов ОБЪЕКТИВНО (без недоверенного судьи покрытия): есть ли связанный коммит-фикс
(граф `Commit−MENTIONS→Task`) и достижим ли он в пуле кандидатов (пре-реранк, как в
`recall_gate`). Классы: `surfaceable` (фикс есть, не достижим — целевой рычаг),
`already_retrieved` (есть и достижим — но ответ провалился → контент под спот-чек), `no_fix`
(связи нет → ingest). Вердикт гейтит surfacing (дёшево) vs ingest (дорого).

Предпосылка R0 (проверяется раннером первым): коммиты должны быть в индексе, иначе
`already_retrieved` структурно недостижим и весь сплит невалиден.

Ядро (этот модуль, U1) — чистые функции. Раннер (`main`, U2) ходит в граф и пул.

Запуск (нужны --extra ml + Neo4j + LLM-ключ):
    uv run --extra ml python -m eval.fix_linkage
"""

from __future__ import annotations

import json

# Предрегистрированные пороги вердикта.
VERDICT_THRESHOLD = 0.50  # доля класса ≥ → его вердикт
_OVERREACH_ROUTES = ("mixed", "multihop")


def classify_linkage(has_fix: bool, fix_reachable: bool) -> str:
    """Класс тикета по (есть связанный коммит-фикс, достижим ли он в пуле кандидатов)."""
    if not has_fix:
        return "no_fix"           # связанного коммита нет → ingest
    if fix_reachable:
        return "already_retrieved"  # есть и достижим, но ответ провалился → контент не решает
    return "surfaceable"          # есть, но не достижим — surfacing поднимет дёшево


def aggregate(classes: list[str], *, threshold: float = VERDICT_THRESHOLD) -> dict:
    """Доли по классам + вердикт направления.

    surfaceable ≥ порога → «surfacing»; no_fix ≥ порога → «ingest»; already_retrieved ≥ порога →
    «content_problem» (коммит достижим, ответ всё равно провалился); иначе «mixed».
    """
    from collections import Counter
    c = Counter(classes)
    n = len(classes)
    frac = {k: c[k] / n for k in ("surfaceable", "already_retrieved", "no_fix")} if n else \
           {"surfaceable": None, "already_retrieved": None, "no_fix": None}

    if not n:
        verdict, reason = "mixed", "нет классифицированных тикетов"
    elif frac["surfaceable"] >= threshold:
        verdict = "surfacing"
        reason = f"surfaceable {frac['surfaceable']:.0%} ≥ {threshold:.0%} — фикс есть, не достаётся; surfacing поднимет"
    elif frac["no_fix"] >= threshold:
        verdict = "ingest"
        reason = f"no_fix {frac['no_fix']:.0%} ≥ {threshold:.0%} — связанного коммита нет; нужен ingest"
    elif frac["already_retrieved"] >= threshold:
        verdict = "content_problem"
        reason = f"already_retrieved {frac['already_retrieved']:.0%} ≥ {threshold:.0%} — фикс достаётся, но ответ провалился; контент не решает"
    else:
        verdict = "mixed"
        reason = f"ни один класс не ≥ {threshold:.0%} (surf {frac['surfaceable']:.0%} / retr {frac['already_retrieved']:.0%} / no_fix {frac['no_fix']:.0%})"

    return {"counts": dict(c), "n": n, "frac": frac, "verdict": verdict, "reason": reason}


# --- Раннер (U2): R0-предпосылка, граф-запрос, достижимость в пуле, спот-чек ---

_COVERAGE = "eval/trial/coverage_diag_results.json"
_REPORT = "eval/trial/fix_linkage_report.md"
_RESULTS = "eval/trial/fix_linkage_results.json"
_SPOTCHECK_N = 4  # сколько текстов коммита показать на класс в спот-чеке


def _fix_commits(conn, task_id: str) -> list[dict]:
    """Связанные коммиты тикета: (c:Commit)-[:MENTIONS]->(t:Task) по t.id. Текст — message."""
    rows = conn.run(
        "MATCH (c:Commit)-[:MENTIONS]->(t:Task {id: $tid}) RETURN c.id AS id, c.message AS message",
        tid=task_id,
    )
    return [{"id": r["id"], "message": r.get("message") or ""} for r in rows]


def main() -> int:
    from pathlib import Path

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.graph import Neo4jConnection
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.recall_gate import candidate_entity_id

    s = load_settings()
    failed = json.loads(Path(_COVERAGE).read_text(encoding="utf-8-sig"))["failed"]
    print(f"fix-linkage: проваленных how-to {len(failed)}", flush=True)

    results: list[dict] = []
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("fix-linkage: Neo4j недоступен — `docker compose up -d`")
            return 1

        # R0: индексированы ли коммиты? Иначе already_retrieved структурно недостижим.
        commit_chunks = conn.run(
            "MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk:commit:' RETURN count(c) AS n"
        )[0]["n"]
        r0_ok = commit_chunks > 0
        print(f"fix-linkage: R0 коммитов-чанков в индексе: {commit_chunks} "
              f"({'OK' if r0_ok else 'НЕТ — замер невалиден'})", flush=True)

        emb = build_embedder(s.embeddings)
        retr = HybridRetriever(
            conn, emb, build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        for i, rec in enumerate(failed, 1):
            sid = rec.get("source_id") or ""
            q = rec["question"]
            try:
                commits = _fix_commits(conn, sid)  # sid = 'task:KAFKA-N' == t.id
                has_fix = len(commits) > 0
                commit_eids = {candidate_entity_id(c["id"]) for c in commits}
                _route, pool = retr._candidate_pool(q)
                pool_eids = {candidate_entity_id(p["id"]) for p in pool}
                fix_reachable = bool(commit_eids & pool_eids)
                # справочно: попал ли фикс в прод-top_k (пост-реранк)
                prod_ids = {candidate_entity_id(c["id"]) for c in retr.retrieve(q).get("candidates", [])}
                in_prod = bool(commit_eids & prod_ids)
                results.append({
                    "source_id": sid, "route": rec.get("route"),
                    "has_fix": has_fix, "fix_reachable": fix_reachable, "in_prod_topk": in_prod,
                    "class": classify_linkage(has_fix, fix_reachable),
                    "commit_msg": commits[0]["message"][:400] if commits else None,
                })
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(failed)}] SKIP: {type(e).__name__}: {e}", flush=True)
            if i % 5 == 0:
                print(f"  [{i}/{len(failed)}]", flush=True)

    v = aggregate([r["class"] for r in results])
    Path(_RESULTS).write_text(json.dumps(
        {"r0_commit_chunks": commit_chunks, "r0_ok": r0_ok, "results": results, "verdict": v},
        ensure_ascii=False, indent=2), encoding="utf-8")

    def _spot(cls):
        picks = [r for r in results if r["class"] == cls and r["commit_msg"]][:_SPOTCHECK_N]
        return [f"  - [{r['source_id']}] {r['commit_msg'][:200]}" for r in picks]

    r0_msg = "OK" if r0_ok else ("**НЕТ: замер невалиден** — коммиты не индексированы, "
                                 "already_retrieved структурно 0")
    lines = [
        "# Замер связи тикет→фикс-коммит",
        "",
        f"**R0 (предпосылка):** коммитов-чанков в индексе {commit_chunks} — {r0_msg}.",
        "",
        f"Проваленных how-to: {v['n']}. Классы: {v['counts']}.",
        f"- доли: surfaceable {_pct(v['frac']['surfaceable'])} · already_retrieved "
        f"{_pct(v['frac']['already_retrieved'])} · no_fix {_pct(v['frac']['no_fix'])}",
        "",
        f"**Вердикт: {v['verdict'].upper()}** — {v['reason']}"
        + ("" if r0_ok else " ⚠️ НЕ доверенный: R0 не прошёл."),
        "",
        "## Спот-чек текстов коммитов (already_retrieved)",
        *(_spot("already_retrieved") or ["  (нет)"]),
        "",
        "## Спот-чек (surfaceable)",
        *(_spot("surfaceable") or ["  (нет)"]),
        "",
        "⚠️ Достижимость — по пре-реранк пулу (`_candidate_pool`); прод-top_k отмечен справочно "
        "(`in_prod_topk`). MENTIONS-полнота зависит от ссылки KAFKA-NNNN в subject коммита.",
    ]
    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE fix-linkage: verdict={v['verdict']} r0_ok={r0_ok} "
          f"surfaceable={_pct(v['frac']['surfaceable'])} -> {_REPORT}", flush=True)
    return 0


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


if __name__ == "__main__":
    raise SystemExit(main())
