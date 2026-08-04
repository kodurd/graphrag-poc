"""Маленький человеческий gold — якорь-правда для калибровки судей.

Согласие двух LLM ≠ истина. Чтобы решить, КТО из судей ближе к человеку (особенно
на honest-refusal), нужен человеческий эталон. Инструмент НЕ угадывает оценки — он
только (1) отбирает ~limit самых ценных для калибровки записей (honest-refusal + где
судьи сильнее всего расходятся), (2) выдаёт лист на разметку, (3) принимает
заполненный человеком JSON и валидирует.

Режимы:
    uv run python -m eval.human_gold build     # отобрать + выдать лист
    uv run python -m eval.human_gold ingest    # проверить заполненный gold_labels.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SNAPSHOT = "eval/trial/quality_snapshot_results.json"
_CROSS = "eval/trial/cross_judge_results.json"
_SHEET = "eval/trial/gold_sheet.md"
_LABELS = "eval/trial/gold_labels.json"
_LIMIT = 40
_METRICS = ("faithfulness", "answer_relevance", "context_precision")

# honest-refusal (фильтр из KTD плана): воздержание ИЛИ отказ-фраза в тексте.
REFUSE_RE = re.compile(
    r"невозможно|недостаточно|отсутству|нет (данных|сведени|информац)|не содержит", re.I
)


def is_refusal(record: dict) -> bool:
    if (record.get("abstained") or {}).get("faithfulness"):
        return True
    return bool(REFUSE_RE.search((record.get("answer") or "")[:400]))


def _max_delta(cross_rec: dict) -> float:
    """Макс |DeepSeek−Qwen| по трём метрикам (насколько судьи расходятся)."""
    ds, qw = cross_rec.get("deepseek", {}), cross_rec.get("qwen", {})
    deltas = [abs(ds.get(m) - qw.get(m)) for m in _METRICS
              if ds.get(m) is not None and qw.get(m) is not None]
    return max(deltas) if deltas else 0.0


def select_gold(
    records: list[dict], cross_results: list[dict] | None = None, *, limit: int = _LIMIT
) -> list[dict]:
    """Отбирает записи для человеческой разметки: все honest-refusal, затем добить
    самыми спорными (макс Δ между судьями) до limit. Дедуп по source_id+question.

    cross_results (из U3) опционально: без него добираем не по Δ, а по порядку.
    """
    def key(r: dict) -> tuple:
        return (r.get("source_id"), r.get("question"))

    picked: dict[tuple, dict] = {}
    for r in records:
        if is_refusal(r) and len(picked) < limit:
            picked[key(r)] = r

    if len(picked) < limit:
        delta_by_key: dict[tuple, float] = {}
        for cr in cross_results or []:
            delta_by_key[(cr.get("source_id"), cr.get("question"))] = _max_delta(cr)
        # оставшиеся, отсортированные по расхождению судей (или 0 без cross)
        rest = [r for r in records if key(r) not in picked]
        rest.sort(key=lambda r: delta_by_key.get(key(r), 0.0), reverse=True)
        for r in rest:
            if len(picked) >= limit:
                break
            picked[key(r)] = r

    return list(picked.values())


def select_nonrefusal_gold(
    records: list[dict], cross_results: list[dict] | None = None, *, limit: int = 15,
    per_route: bool = True,
) -> list[dict]:
    """Отбирает НЕ-отказы для глобальной калибровки судьи (не только на refusal).

    Приоритет — записи, где судьи сильнее расходятся (макс Δ из cross): там человеческий
    якорь ценнее. При per_route — раскладывает лимит по маршрутам (mixed/multihop/factual),
    чтобы доверие проверялось на разных типах, а не только на самом частом.
    """
    def key(r: dict) -> tuple:
        return (r.get("source_id"), r.get("question"))

    delta_by_key: dict[tuple, float] = {}
    for cr in cross_results or []:
        delta_by_key[(cr.get("source_id"), cr.get("question"))] = _max_delta(cr)

    cands = [r for r in records if not is_refusal(r)]
    cands.sort(key=lambda r: delta_by_key.get(key(r), 0.0), reverse=True)

    if not per_route:
        return cands[:limit]

    # Round-robin по маршрутам: берём по очереди из каждого маршрута (внутри — по Δ).
    from collections import defaultdict, deque
    buckets: dict[str, deque] = defaultdict(deque)
    for r in cands:
        buckets[r.get("route") or "?"].append(r)
    picked: list[dict] = []
    routes = list(buckets)
    while len(picked) < limit and any(buckets[rt] for rt in routes):
        for rt in routes:
            if buckets[rt] and len(picked) < limit:
                picked.append(buckets[rt].popleft())
    return picked


def render_sheet(selected: list[dict]) -> str:
    """Лист для человека: вопрос + ответ + (усечённый) контекст + пустые поля оценок."""
    lines = [
        "# Человеческий gold — разметка",
        "",
        "Оцени КАЖДУЮ метрику 0..1 и впиши в `eval/trial/gold_labels.json` "
        "(ключ = source_id). Метрики: faithfulness (не выдумал), answer_relevance "
        "(в тему), context_precision (контекст релевантен).",
        "",
    ]
    for i, r in enumerate(selected, 1):
        ctx = " | ".join((r.get("context_texts") or [])[:3])[:500]
        lines += [
            f"## {i}. {r.get('source_id')}  {'[REFUSAL]' if is_refusal(r) else ''}",
            f"**Вопрос:** {r.get('question')}",
            f"**Ответ:** {(r.get('answer') or '')[:600]}",
            f"**Контекст (фрагменты):** {ctx}",
            "",
        ]
    return "\n".join(lines)


def load_gold(path: str = _LABELS) -> dict:
    """Читает и валидирует заполненный человеком gold. Формат:
    {"task:KAFKA-N": {"faithfulness":0..1, "answer_relevance":0..1, "context_precision":0..1}}"""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not data:
        raise ValueError("gold пуст или не объект {source_id: {метрики}}")
    for sid, scores in data.items():
        for m in _METRICS:
            v = (scores or {}).get(m)
            if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                raise ValueError(f"{sid}: метрика {m} должна быть числом 0..1, дано {v!r}")
    return data


def main() -> int:
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"

    if mode == "build":
        records = json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]
        cross = None
        if Path(_CROSS).exists():
            cross = json.loads(Path(_CROSS).read_text(encoding="utf-8-sig"))["records"]
        selected = select_gold(records, cross)
        Path(_SHEET).write_text(render_sheet(selected), encoding="utf-8")
        # скелет для заполнения
        skel = {r.get("source_id"): {m: None for m in _METRICS} for r in selected}
        Path(_LABELS + ".template").write_text(
            json.dumps(skel, ensure_ascii=False, indent=2), encoding="utf-8")
        refusals = sum(1 for r in selected if is_refusal(r))
        print(f"build: отобрано {len(selected)} (refusal {refusals}) -> {_SHEET}; "
              f"скелет -> {_LABELS}.template (заполни и сохрани как {_LABELS})")
        return 0

    if mode == "build_nonrefusal":
        # Глобальное доверие: добавить ~15 НЕ-отказов к существующим отказам.
        records = json.loads(Path(_SNAPSHOT).read_text(encoding="utf-8-sig"))["records"]
        cross = None
        if Path(_CROSS).exists():
            cross = json.loads(Path(_CROSS).read_text(encoding="utf-8-sig"))["records"]
        new = select_nonrefusal_gold(records, cross, limit=15)
        # Лист = уже размеченные (28) + новые не-отказы → UI покажет прогресс и пустые новые.
        done = load_gold(_LABELS) if Path(_LABELS).exists() else {}
        by_sid = {r.get("source_id"): r for r in records}
        selected = [by_sid[sid] for sid in done if sid in by_sid] + new
        Path(_SHEET).write_text(render_sheet(selected), encoding="utf-8")
        skel = {**{sid: {m: None for m in _METRICS} for sid in done},
                **{r.get("source_id"): {m: None for m in _METRICS} for r in new}}
        Path(_LABELS + ".template").write_text(
            json.dumps(skel, ensure_ascii=False, indent=2), encoding="utf-8")
        from collections import Counter
        print(f"build_nonrefusal: +{len(new)} не-отказов "
              f"(маршруты {dict(Counter(r.get('route') for r in new))}); "
              f"лист {len(selected)} (из них {len(done)} уже размечены) -> {_SHEET}; "
              f"template -> {_LABELS}.template. Размечай новые через eval.gold_ui.")
        return 0

    if mode == "ingest":
        gold = load_gold()
        print(f"ingest: gold валиден, {len(gold)} размеченных записей ({_LABELS})")
        return 0

    print(f"human-gold: режим {mode!r} неизвестен (build|build_nonrefusal|ingest)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
