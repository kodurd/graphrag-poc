"""Сборка отчёта по результатам прогона качества.

Чистые функции над записями из `quality_eval`. Ключевые решения:
- по каждой метрике даётся не только среднее, но и **распределение** (квантили и
  гистограмма по корзинам) — одинокое среднее вводит в заблуждение;
- значения `None` (сбой судьи) исключаются из знаменателя, а метрика, где
  оценок не осталось, помечается «не оценено», а не нулём;
- retrieval P/R/F1 рендерится **отдельной секцией**: это другая популяция
  вопросов (графовый golden set), её нельзя смешивать с per-question метриками;
- в отчёт включается оговорка про само-оценку (одна модель генерит, судит и
  размечает).
"""

from __future__ import annotations

METRIC_TITLES = {
    "faithfulness": "Faithfulness (не выдумывает)",
    "answer_relevance": "Answer relevance (отвечает по делу)",
    "context_precision": "Context precision (retrieval релевантен)",
    "answer_correctness": "Answer correctness (совпадает с эталоном)",
    "context_recall": "Context recall (эталон покрыт контекстом)",
}

SELF_GRADING_CAVEAT = (
    "⚠️ Само-оценка: одна и та же модель генерирует ответы, судит их и пишет "
    "эталоны для размеченного среза. Оценки оптимистичны и ограничены "
    "надёжностью самой модели — это потолок метода, а не абсолютная истина."
)

_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def _scored(records: list[dict], metric: str) -> list[float]:
    """Значения метрики без `None` (сбой судьи не занижает агрегат)."""
    out = []
    for r in records:
        v = (r.get("metrics") or {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def _quantile(values: list[float], q: float) -> float:
    """Квантиль по ближайшему рангу (список должен быть отсортирован)."""
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def summarize_metric(records: list[dict], metric: str) -> dict:
    """Агрегат по метрике: n, доля оценённых, среднее, квантили, гистограмма."""
    values = sorted(_scored(records, metric))
    total = len(records)
    if not values:
        return {"metric": metric, "n": 0, "total": total, "scored": False}

    histogram = {
        f"{lo:.1f}-{hi if hi <= 1 else 1.0:.1f}": sum(1 for v in values if lo <= v < hi)
        for lo, hi in _BUCKETS
    }
    return {
        "metric": metric,
        "n": len(values),
        "total": total,
        "scored": True,
        "mean": sum(values) / len(values),
        "p50": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "p90": _quantile(values, 0.9),
        "histogram": histogram,
    }


def breakdown(records: list[dict], metric: str, key: str) -> dict:
    """Среднее метрики в разрезе поля записи (`route`) — только оценённые."""
    groups: dict[str, list[float]] = {}
    for r in records:
        v = (r.get("metrics") or {}).get(metric)
        if v is None:
            continue
        groups.setdefault(str(r.get(key)), []).append(float(v))
    return {g: sum(vals) / len(vals) for g, vals in sorted(groups.items())}


def source_type(record: dict) -> str:
    """Тип источника из префикса id: 'task:KAFKA-1' -> Task, 'page:1' -> Page."""
    sid = str(record.get("source_id") or "")
    prefix = sid.split(":", 1)[0].lower()
    return {"task": "Task", "page": "Page"}.get(prefix, "unknown")


def breakdown_by_source(records: list[dict], metric: str) -> dict:
    """Среднее метрики по типу источника (Task/Page)."""
    groups: dict[str, list[float]] = {}
    for r in records:
        v = (r.get("metrics") or {}).get(metric)
        if v is None:
            continue
        groups.setdefault(source_type(r), []).append(float(v))
    return {g: sum(vals) / len(vals) for g, vals in sorted(groups.items())}


def abstention_stats(records: list[dict], metric: str = "faithfulness") -> dict:
    """Доля воздержаний и сбоев по метрике (различает N/A-воздержание от сбоя судьи).

    Воздержание — `record["abstained"][metric] is True` (score None по замыслу).
    Сбой судьи — score None БЕЗ флага воздержания. Оценено — score не None.
    """
    total = len(records)
    abstained = failed = scored = 0
    for r in records:
        v = (r.get("metrics") or {}).get(metric)
        is_abstained = bool((r.get("abstained") or {}).get(metric))
        if is_abstained:
            abstained += 1
        elif v is None:
            failed += 1
        else:
            scored += 1
    return {
        "total": total,
        "abstained": abstained,
        "failed": failed,
        "scored": scored,
        "abstention_rate": abstained / total if total else 0.0,
    }


def failures(records: list[dict], metric: str, *, threshold: float = 0.5, top: int = 5) -> list[dict]:
    """Примеры провалов: оценённые записи ниже порога, худшие первыми."""
    scored = [
        r for r in records if (r.get("metrics") or {}).get(metric) is not None
        and float(r["metrics"][metric]) < threshold
    ]
    scored.sort(key=lambda r: float(r["metrics"][metric]))
    return scored[:top]


def render_report(
    results: dict,
    retrieval: dict | None = None,
    *,
    threshold: float = 0.5,
) -> str:
    """Markdown-отчёт: агрегаты, распределения, разбивки, примеры провалов."""
    records = results.get("records", [])
    counts = results.get("counts", {})
    lines = [
        "# Отчёт: качество ответов RAG",
        "",
        SELF_GRADING_CAVEAT,
        "",
        f"Вопросов: {counts.get('questions', 0)} · размеченный срез: "
        f"{counts.get('labeled', 0)} · всего записей: {counts.get('total', len(records))}",
        "",
        "## Метрики",
        "",
    ]

    for metric in METRIC_TITLES:
        summary = summarize_metric(records, metric)
        title = METRIC_TITLES[metric]

        def _abstention_line() -> list[str]:
            # faithfulness-специфично: доля воздержаний рядом со средним, отдельно
            # от сбоев — чтобы рост среднего (за счёт исключения) не маскировал
            # «RAG стал воздерживаться чаще».
            if metric != "faithfulness":
                return []
            a = abstention_stats(records, metric)
            return [
                f"- воздержаний {a['abstained']}/{a['total']} "
                f"({a['abstention_rate']:.1%}) · сбоев судьи {a['failed']} "
                "(воздержание ≠ сбой)"
            ]

        if not summary["scored"]:
            lines += [f"### {title}", "", "_не оценено_ (нет успешных оценок судьи)"]
            lines += _abstention_line()
            lines.append("")
            continue
        lines += [
            f"### {title}",
            "",
            f"- среднее **{summary['mean']:.3f}** · p10 {summary['p10']:.2f} · "
            f"p50 {summary['p50']:.2f} · p90 {summary['p90']:.2f}",
            f"- оценено {summary['n']} из {summary['total']} записей",
        ]
        lines += _abstention_line()
        lines += [f"- распределение: {summary['histogram']}"]
        by_route = breakdown(records, metric, "route")
        if by_route:
            lines.append(
                "- по маршруту: "
                + ", ".join(f"{k} {v:.3f}" for k, v in by_route.items())
            )
        by_source = breakdown_by_source(records, metric)
        if by_source:
            lines.append(
                "- по типу источника: "
                + ", ".join(f"{k} {v:.3f}" for k, v in by_source.items())
            )
        lines.append("")

    lines += ["## Примеры провалов", ""]
    any_failure = False
    for metric in ("faithfulness", "answer_relevance"):
        bad = failures(records, metric, threshold=threshold)
        if not bad:
            continue
        any_failure = True
        lines.append(f"### {METRIC_TITLES[metric]} < {threshold}")
        lines.append("")
        for r in bad:
            lines += [
                f"- **{r.get('question')}** — {metric} "
                f"{r['metrics'][metric]:.2f}, маршрут {r.get('route')}",
                f"  источники: {r.get('citations') or r.get('context_ids')}",
            ]
        lines.append("")
    if not any_failure:
        lines += ["_нет записей ниже порога_", ""]

    if retrieval:
        lines += [
            "## Retrieval P/R/F1 (отдельная популяция)",
            "",
            "Считается на графовом golden set (`build_from_graph`), а не на "
            "авто-сгенерированных вопросах — у тех нет эталонных множеств узлов. "
            "Поэтому цифры не сопоставимы напрямую с метриками выше.",
            "",
            f"- n={retrieval.get('n')} · precision {retrieval.get('precision', 0):.3f} "
            f"· recall {retrieval.get('recall', 0):.3f} · f1 {retrieval.get('f1', 0):.3f}",
            "",
        ]

    return "\n".join(lines)
