"""Анализ судей: согласие как КОНСИСТЕНТНОСТЬ + калибровка против человеческого gold.

Центральный принцип (из ревью плана): согласие двух LLM ≠ истина — две модели делят
слепые пятна. Поэтому:
- согласие DeepSeek↔Qwen подаём как консистентность (распределение Δ, корреляция,
  доля |Δ|≤порог со СВИПОМ порога, не одним числом);
- КТО прав (особенно на honest-refusal) решает только человеческий gold — по MAE к
  человеку.

Запуск: uv run python -m eval.judge_calibration
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.human_gold import is_refusal, load_gold

_CROSS = "eval/trial/cross_judge_results.json"
_GOLD = "eval/trial/gold_labels.json"
_REPORT = "eval/trial/judge_calibration_report.md"
_METRICS = ("faithfulness", "answer_relevance", "context_precision")
_THRESHOLDS = (0.1, 0.2, 0.3)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def mae(pred, gold):
    """Средняя |pred−gold| по парам, где оба не None."""
    ds = [abs(p - g) for p, g in zip(pred, gold) if p is not None and g is not None]
    return _mean(ds)


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x, _ in pairs) ** 0.5
    dy = sum((y - my) ** 2 for _, y in pairs) ** 0.5
    return num / (dx * dy) if dx and dy else None


def consistency(cross: list[dict]) -> dict:
    """Согласие судей как консистентность: по метрике — корреляция + доля |Δ|≤порог."""
    out = {}
    for m in _METRICS:
        ds = [r.get("deepseek", {}).get(m) for r in cross]
        qw = [r.get("qwen", {}).get(m) for r in cross]
        deltas = [abs(a - b) for a, b in zip(ds, qw) if a is not None and b is not None]
        out[m] = {
            "n": len(deltas),
            "corr": pearson(ds, qw),
            "mean_abs_delta": _mean(deltas),
            "agree": {f"|d|<={t}": (sum(1 for d in deltas if d <= t) / len(deltas)
                                    if deltas else None) for t in _THRESHOLDS},
        }
    return out


def calibration(cross: list[dict], gold: dict) -> dict:
    """MAE каждого судьи к человеку на gold-подвыборке, по метрике."""
    rows = [r for r in cross if r.get("source_id") in gold]
    out = {"n": len(rows)}
    for m in _METRICS:
        g = [gold[r["source_id"]].get(m) for r in rows]
        out[m] = {
            "deepseek_mae": mae([r.get("deepseek", {}).get(m) for r in rows], g),
            "qwen_mae": mae([r.get("qwen", {}).get(m) for r in rows], g),
        }
    return out


def refusal_verdict(cross: list[dict], gold: dict) -> dict:
    """На honest-refusal с gold: кто из судей ближе к человеку по relevance.

    Прямой тест гипотезы «relevance занижен само-судьёй» — но через ЯКОРЬ (gold),
    а не через «Qwen выше DeepSeek» (что неразрешимо двумя судьями)."""
    rows = [r for r in cross if is_refusal(r) and r.get("source_id") in gold]
    g = [gold[r["source_id"]].get("answer_relevance") for r in rows]
    ds_mae = mae([r.get("deepseek", {}).get("answer_relevance") for r in rows], g)
    qw_mae = mae([r.get("qwen", {}).get("answer_relevance") for r in rows], g)
    verdict = "недостаточно gold-отказов"
    if ds_mae is not None and qw_mae is not None:
        if abs(ds_mae - qw_mae) < 0.02:
            verdict = "судьи одинаково близки к человеку на отказах"
        elif qw_mae < ds_mae:
            verdict = "Qwen ближе к человеку — DeepSeek занижал relevance на честных отказах (гипотеза подтверждена)"
        else:
            verdict = "DeepSeek ближе к человеку — гипотеза про занижение НЕ подтверждена"
    return {"n_refusal_gold": len(rows), "deepseek_relevance_mae": ds_mae,
            "qwen_relevance_mae": qw_mae, "verdict": verdict}


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main() -> int:
    if not Path(_CROSS).exists():
        print(f"calibration: нет {_CROSS} — сначала eval.cross_judge (нужен Qwen-ключ)")
        return 1
    cross = json.loads(Path(_CROSS).read_text(encoding="utf-8-sig"))["records"]
    gold = load_gold(_GOLD) if Path(_GOLD).exists() else {}

    cons = consistency(cross)
    lines = [
        "# Калибровка судей: консистентность + правда-через-gold",
        "",
        "> ⚠️ **Согласие двух LLM = консистентность, НЕ доказательство правильности.** "
        "Две модели могут разделять слепое пятно. Абсолютную правоту решает человеческий gold.",
        "",
        f"Записей: {len(cross)} · gold-размечено: {len(gold)}",
        "",
        "## Консистентность DeepSeek ↔ Qwen (не истина!)",
        "",
        "| метрика | n | корреляция | ср.|Δ| | доля |Δ|≤0.1/0.2/0.3 |",
        "|---|---|---|---|---|",
    ]
    for m in _METRICS:
        c = cons[m]
        ag = c["agree"]
        lines.append(
            f"| {m} | {c['n']} | {_fmt(c['corr'])} | {_fmt(c['mean_abs_delta'])} | "
            f"{_fmt(ag['|d|<=0.1'])} / {_fmt(ag['|d|<=0.2'])} / {_fmt(ag['|d|<=0.3'])} |")

    if gold:
        cal = calibration(cross, gold)
        rv = refusal_verdict(cross, gold)
        lines += [
            "", f"## Калибровка против человека (gold n={cal['n']})", "",
            "MAE к человеку — меньше = ближе к правде.", "",
            "| метрика | DeepSeek MAE | Qwen MAE |", "|---|---|---|",
        ]
        for m in _METRICS:
            lines.append(f"| {m} | {_fmt(cal[m]['deepseek_mae'])} | {_fmt(cal[m]['qwen_mae'])} |")
        lines += [
            "", "## Вердикт honest-refusal (через gold, не через согласие)", "",
            f"- gold-отказов: {rv['n_refusal_gold']}",
            f"- relevance MAE к человеку: DeepSeek {_fmt(rv['deepseek_relevance_mae'])} · "
            f"Qwen {_fmt(rv['qwen_relevance_mae'])}",
            f"- **{rv['verdict']}**",
        ]
    else:
        lines += ["", "## Калибровка против человека", "",
                  "_gold не размечен_ — запусти `eval.human_gold build`, размечь, "
                  "положи `gold_labels.json`, перезапусти. Без gold правоту судей "
                  "определить НЕЛЬЗЯ (только консистентность выше)."]

    Path(_REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE calibration -> {_REPORT} (gold={'да' if gold else 'нет'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
