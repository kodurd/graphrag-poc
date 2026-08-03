"""Диагностика шума faithfulness-судьи.

Берёт зафиксированные ответы из снимка качества (`quality_snapshot_results.json`) и
пере-судит их N раз, варьируя ТОЛЬКО оценку судьи (не генерацию), чтобы отделить
дисперсию судьи от реальной faithfulness. Вердикт по предрегистрированному порогу:
нужен ли постоянный мульти-сэмпл судья.

Ядро (этот модуль, U1) — чистые функции без I/O: отбор групп, статистика стабильности,
вердикт. Раннер (`main`, U2) реконструирует контекст пере-извлечением, судит N раз и
пишет отчёт.

Запуск (нужны --extra ml + Neo4j с корпусом + LLM-ключ):
    uv run --extra ml python -m eval.judge_noise
"""

from __future__ import annotations

from eval.quality_report import _BUCKETS

# Пороги вердикта — предрегистрированы (см. план/origin), не подгоняются под итог.
NOISY_FRAC_THRESHOLD = 0.30  # доля нестабильных записей полюса «0»
LOW_MEAN_THRESHOLD = 0.30  # macro-усреднённая оценка полюса «0» (подъём над записанным ≈0)
N_SAMPLES = 5


def _bucket_of(score: float) -> int:
    """Индекс корзины faithfulness (как в quality_report._BUCKETS)."""
    for i, (lo, hi) in enumerate(_BUCKETS):
        if lo <= score < hi:
            return i
    return len(_BUCKETS) - 1  # score вне диапазона (клампится в последнюю)


def select_groups(records: list[dict]) -> dict[str, list[dict]]:
    """Разбивает записи снимка на группы по faithfulness: low<0.2 / high>=0.8 / mid.

    Записи без оценки (faithfulness None) отбрасываются. На двухполюсных данных n=96
    `mid` выходит пустым — это допустимо, не ошибка.
    """
    groups: dict[str, list[dict]] = {"low": [], "high": [], "mid": []}
    for r in records:
        v = (r.get("metrics") or {}).get("faithfulness")
        if v is None:
            continue
        v = float(v)
        if v < 0.2:
            groups["low"].append(r)
        elif v >= 0.8:
            groups["high"].append(r)
        else:
            groups["mid"].append(r)
    return groups


def stability(scores: list[float | None]) -> dict:
    """Статистика по N оценкам одной записи.

    `scores` — список из N элементов, каждый число либо None (воздержание/сбой судьи).
    Все N None → запись `no_score` (усреднённая None, не stable, не «шумная»). Иначе:
    усреднённая по не-None, дисперсия по не-None, `stable` = все не-None в одной корзине.
    """
    non_none = [float(s) for s in scores if s is not None]
    if not non_none:
        return {"mean": None, "variance": None, "stable": False,
                "n_scored": 0, "no_score": True}
    mean = sum(non_none) / len(non_none)
    variance = sum((s - mean) ** 2 for s in non_none) / len(non_none)
    stable = len({_bucket_of(s) for s in non_none}) == 1
    return {"mean": mean, "variance": variance, "stable": stable,
            "n_scored": len(non_none), "no_score": False}


def group_stats(per_record: list[dict]) -> dict:
    """Агрегат по группе из per-record статистик `stability`.

    `no_score`-записи исключены из знаменателя доли шумных и из усреднений. Пустая
    (или полностью `no_score`) группа → структура с пометкой «n/a», без падения.
    Усреднённая faithfulness — **macro**: среднее по-записных средних.
    """
    scored = [r for r in per_record if not r["no_score"]]
    no_score = sum(1 for r in per_record if r["no_score"])
    if not scored:
        return {"n": len(per_record), "n_scored": 0, "no_score": no_score,
                "noisy_frac": None, "mean_variance": None, "mean_faith": None,
                "note": "n/a"}
    noisy = [r for r in scored if not r["stable"]]
    return {
        "n": len(per_record),
        "n_scored": len(scored),
        "no_score": no_score,
        "noisy_frac": len(noisy) / len(scored),
        "mean_variance": sum(r["variance"] for r in scored) / len(scored),
        "mean_faith": sum(r["mean"] for r in scored) / len(scored),  # macro
    }


def verdict(
    low_stats: dict,
    *,
    noisy_frac_threshold: float = NOISY_FRAC_THRESHOLD,
    low_mean_threshold: float = LOW_MEAN_THRESHOLD,
) -> dict:
    """Предрегистрированный вердикт fix/no-fix по статистике полюса «0».

    Фикс (мульти-сэмпл судья) оправдан, если выполнено любое:
      1) нестабильность — доля шумных в low ≥ порога; ИЛИ
      2) систематическое занижение — macro-усреднённая оценка полюса «0» ≥ порога
         (полюс пере-усреднился заметно выше своего записанного ≈0).
    Сравнения с общей 0.59 нет: полюс отобран как <0.2, такой сдвиг сработал бы всегда.
    """
    if low_stats.get("n_scored", 0) == 0 or low_stats.get("noisy_frac") is None:
        return {"fix_warranted": False, "reason": "нет оценённых записей в полюсе «0» — вердикт невозможен",
                "noisy_frac": low_stats.get("noisy_frac"), "low_mean": low_stats.get("mean_faith")}

    noisy_frac = low_stats["noisy_frac"]
    low_mean = low_stats["mean_faith"]
    gate_noisy = noisy_frac >= noisy_frac_threshold
    # falsy-zero: low_mean ровно 0.0 — валидное число, не «нет данных»
    gate_lift = low_mean is not None and low_mean >= low_mean_threshold
    fix = gate_noisy or gate_lift

    if not fix:
        reason = (f"судья стабилен: доля шумных {noisy_frac:.0%} < {noisy_frac_threshold:.0%} "
                  f"и подъём полюса {low_mean:.3f} < {low_mean_threshold:.2f} — метрики честны")
    else:
        parts = []
        if gate_noisy:
            parts.append(f"доля шумных {noisy_frac:.0%} ≥ {noisy_frac_threshold:.0%}")
        if gate_lift:
            parts.append(f"подъём полюса {low_mean:.3f} ≥ {low_mean_threshold:.2f}")
        reason = "нужен мульти-сэмпл судья: " + "; ".join(parts)

    return {"fix_warranted": fix, "reason": reason,
            "noisy_frac": noisy_frac, "low_mean": low_mean,
            "gate_noisy": gate_noisy, "gate_lift": gate_lift}


# --- Раннер (U2): реконструкция контекста, N пере-оценок, отчёт ---

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_REPORT = "eval/trial/judge_noise_report.md"
_RESULTS = "eval/trial/judge_noise_results.json"
_CONTROL_SIZE = 15  # подвыборка полюса «≈1» для контроля (первые N, детерминировано)


def _rejudge_record(retr, llm, record: dict, judge_fn) -> dict:
    """Реконструирует контекст по вопросу и судит ЗАФИКСИРОВАННЫЙ ответ N раз."""
    from graphrag.generate.answer import build_context

    retrieved = retr.retrieve(record["question"])
    ctx = build_context(retrieved.get("candidates", []))
    ctx_texts = [c.text for c in ctx]
    scores: list[float | None] = []
    abstained_n = 0
    for _ in range(N_SAMPLES):
        score, abstained = judge_fn(llm, record["answer"], ctx_texts)
        scores.append(score)
        if abstained:
            abstained_n += 1
    st = stability(scores)
    return {"question": record["question"], "source_id": record.get("source_id"),
            "route": record.get("route"),
            "recorded_faith": (record.get("metrics") or {}).get("faithfulness"),
            "scores": scores, "abstained_n": abstained_n, **st}


def main() -> int:
    import json
    from pathlib import Path

    from graphrag.config import load_settings
    from graphrag.embeddings import build_embedder, build_reranker
    from graphrag.graph import Neo4jConnection
    from graphrag.llm import build_llm
    from graphrag.retrieval.hybrid import HybridRetriever

    from eval.metrics import judge_faithfulness

    s = load_settings()
    if s.llm.provider == "api" and not s.llm.api_key:
        print("judge-noise: не задан LLM_API_KEY (.env).")
        return 1

    records = json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]
    groups = select_groups(records)
    # Отбор: полюс «0» — все; контроль «≈1» — первые _CONTROL_SIZE; mid — если есть (обычно пуст).
    selected = {
        "low": groups["low"],
        "high": groups["high"][:_CONTROL_SIZE],
        "mid": groups["mid"][:_CONTROL_SIZE],
    }
    print(
        f"judge-noise: low={len(selected['low'])} high(control)={len(selected['high'])} "
        f"mid={len(selected['mid'])} · N={N_SAMPLES} · reranker {s.reranker.provider}",
        flush=True,
    )

    per_group: dict[str, list[dict]] = {"low": [], "high": [], "mid": []}
    with Neo4jConnection(s.neo4j) as conn:
        if not conn.verify_connectivity():
            print("judge-noise: Neo4j недоступен — `docker compose up -d`")
            return 1
        retr = HybridRetriever(
            conn, build_embedder(s.embeddings), build_reranker(s.reranker),
            top_k=s.retrieval.top_k, rerank_top_k=s.retrieval.rerank_top_k,
            max_hops=s.retrieval.max_hops, min_rerank_score=s.retrieval.min_rerank_score,
        )
        llm = build_llm(s.llm, role="generation")
        for gname, recs in selected.items():
            for i, record in enumerate(recs, 1):
                try:
                    per_group[gname].append(
                        _rejudge_record(retr, llm, record, judge_faithfulness)
                    )
                except Exception as e:  # noqa: BLE001 — сбой на записи не роняет прогон
                    print(f"  [{gname} {i}/{len(recs)}] SKIP: {type(e).__name__}: {e}", flush=True)
                    continue
                if (len(per_group[gname])) % 5 == 0:
                    print(f"  [{gname}] обработано {len(per_group[gname])}/{len(recs)}", flush=True)

    stats = {g: group_stats(rs) for g, rs in per_group.items()}
    v = verdict(stats["low"])

    Path(_RESULTS).write_text(
        json.dumps({"per_group": per_group, "group_stats": stats, "verdict": v},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(stats, v, per_group)
    print(
        f"DONE judge-noise: low noisy={_pct(stats['low'].get('noisy_frac'))} "
        f"low_mean={_fmt(stats['low'].get('mean_faith'))} -> "
        f"{'FIX' if v['fix_warranted'] else 'NO-FIX'} ({v['reason']}) -> {_REPORT}",
        flush=True,
    )
    return 0


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def _write_report(stats: dict, v: dict, per_group: dict) -> None:
    from pathlib import Path

    lines = [
        "# Диагностика шума faithfulness-судьи",
        "",
        f"Каждый зафиксированный ответ из снимка пере-судён N={N_SAMPLES} раз (варьируется только "
        "оценка судьи, не генерация). Контекст реконструирован пере-извлечением (cross-encoder).",
        "",
        "## По группам",
        "",
        "| Группа | n | оценено | no_score | доля шумных | ср. дисперсия | усреднённая faith |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {"low": "полюс «≈0»", "high": "контроль «≈1»", "mid": "середина"}
    for g in ("low", "high", "mid"):
        st = stats[g]
        lines.append(
            f"| {labels[g]} | {st['n']} | {st['n_scored']} | {st['no_score']} | "
            f"{_pct(st.get('noisy_frac'))} | {_fmt(st.get('mean_variance'))} | "
            f"{_fmt(st.get('mean_faith'))} |"
        )
    lines += [
        "",
        "## Вердикт (предрегистрированный порог)",
        "",
        f"- доля шумных в полюсе «0»: {_pct(v.get('noisy_frac'))} "
        f"(порог {NOISY_FRAC_THRESHOLD:.0%})",
        f"- macro-усреднённая оценка полюса «0»: {_fmt(v.get('low_mean'))} "
        f"(порог {LOW_MEAN_THRESHOLD:.2f}; подъём над записанным ≈0)",
        "",
        f"**Решение: {'НУЖЕН ФИКС (мульти-сэмпл судья)' if v['fix_warranted'] else 'МЕТРИКИ ЧЕСТНЫ'}** "
        f"— {v['reason']}",
        "",
        "⚠️ Реконструированный контекст фиксирован по N оценкам одной записи (замер дисперсии судьи "
        "корректен), но может отличаться от того, что видел исходный ответ. N мал — вердикт "
        "бинарный, при пограничности увеличить N. Само-судья.",
    ]
    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
