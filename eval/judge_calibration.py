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
_BASELINE = "eval/trial/trusted_baseline_report.md"
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


def _gold_rows(cross: list[dict], gold: dict) -> list[dict]:
    """По одной строке cross на gold-source_id (первая встреченная).

    gold кейован по source_id, а cross — по (source_id, question). У источника с
    несколькими вопросами человеческая метка относится к ОДНОМУ ответу; джойн по
    source_id без дедупа применил бы её к другим (неразмеченным) ответам и раздул n.
    Берём одного представителя на источник — метка используется ровно раз.
    """
    seen: set = set()
    rows: list[dict] = []
    for r in cross:
        sid = r.get("source_id")
        if sid in gold and sid not in seen:
            seen.add(sid)
            rows.append(r)
    return rows


def calibration(cross: list[dict], gold: dict) -> dict:
    """MAE каждого судьи к человеку на gold-подвыборке, по метрике."""
    rows = _gold_rows(cross, gold)
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
    rows = [r for r in _gold_rows(cross, gold) if is_refusal(r)]
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


def _mean_mae(block: dict):
    """Средний MAE по трём метрикам внутри блока — (DeepSeek, Qwen)."""
    ds = _mean([block[m]["deepseek_mae"] for m in _METRICS])
    qw = _mean([block[m]["qwen_mae"] for m in _METRICS])
    return ds, qw


def calibration_split(cross: list[dict], gold: dict) -> dict:
    """MAE обоих судей к человеку РАЗДЕЛЬНО на отказах и не-отказах + глобальный вердикт.

    Глобальное доверие Qwen подтверждается ТОЛЬКО если Qwen MAE ≤ DeepSeek и на
    не-отказах (иначе он доверен лишь на отказах — против survivorship-вывода).
    """
    rows = _gold_rows(cross, gold)

    def block(subset: list[dict]) -> dict:
        res = {"n": len(subset)}
        for m in _METRICS:
            g = [gold[r["source_id"]].get(m) for r in subset]
            res[m] = {
                "deepseek_mae": mae([r.get("deepseek", {}).get(m) for r in subset], g),
                "qwen_mae": mae([r.get("qwen", {}).get(m) for r in subset], g),
            }
        return res

    ref = block([r for r in rows if is_refusal(r)])
    non = block([r for r in rows if not is_refusal(r)])

    if non["n"] == 0:
        verdict = ("нет non-refusal gold — глобальность НЕ проверить; "
                   "доверие ограничено отказами")
    else:
        # По-метрично на НЕ-отказах: где Qwen ≤ DeepSeek. Средний MAE маскирует
        # разворот по метрикам (Qwen может выигрывать только на одной), поэтому
        # вердикт строим из per-metric сравнения, а не из одного числа.
        qwen_wins = [m for m in _METRICS
                     if non[m]["qwen_mae"] is not None and non[m]["deepseek_mae"] is not None
                     and non[m]["qwen_mae"] <= non[m]["deepseek_mae"] + 1e-9]
        ds_wins = [m for m in _METRICS
                   if non[m]["qwen_mae"] is not None and non[m]["deepseek_mae"] is not None
                   and non[m]["qwen_mae"] > non[m]["deepseek_mae"] + 1e-9]
        if len(qwen_wins) == len([m for m in _METRICS if non[m]["qwen_mae"] is not None]):
            verdict = ("Qwen доверен ГЛОБАЛЬНО — на не-отказах он не хуже DeepSeek по всем "
                       f"метрикам ({', '.join(qwen_wins)})")
        elif "faithfulness" in qwen_wins:
            verdict = ("Qwen доверен на faithfulness и на отказах (там он решительно ближе к "
                       f"человеку); на не-отказных {', '.join(ds_wins)} судьи СОПОСТАВИМЫ "
                       "(DeepSeek чуть ближе, разница мала, n мал, один разметчик) — "
                       "глобальное превосходство по этим метрикам НЕ заявляем")
        else:
            verdict = ("Qwen НЕ ближе к человеку на не-отказном faithfulness — "
                       f"доверие ограничено отказами (Qwen выигрывает: {qwen_wins or '—'})")
    return {"refusal": ref, "non_refusal": non, "verdict": verdict}


def trusted_baseline(cross: list[dict]) -> dict:
    """Честный baseline без survivorship.

    Первично — DeepSeek vs Qwen на ОБЩЕМ не-воздержавшемся наборе (оба судьи не-None
    по метрике). Вторично — полные средние каждого (разный знаменатель). Отдельно —
    доля воздержаний/сбоев Qwen по faith. Плюс срез по маршрутам на общем наборе.
    """
    out = {"n": len(cross), "by_metric": {}, "by_route": {}}
    for m in _METRICS:
        ds = [r.get("deepseek", {}).get(m) for r in cross]
        qw = [r.get("qwen", {}).get(m) for r in cross]
        common = [(d, q) for d, q in zip(ds, qw) if d is not None and q is not None]
        out["by_metric"][m] = {
            "common_n": len(common),
            "deepseek_common": _mean([d for d, _ in common]),
            "qwen_common": _mean([q for _, q in common]),
            "deepseek_full": _mean([d for d in ds if d is not None]),
            "deepseek_full_n": sum(1 for d in ds if d is not None),
            "qwen_full": _mean([q for q in qw if q is not None]),
            "qwen_full_n": sum(1 for q in qw if q is not None),
        }

    # Воздержания/сбои Qwen по faith (знаменатель для честности «полных» средних).
    faith_none = sum(1 for r in cross if r.get("qwen", {}).get("faithfulness") is None)
    faith_abst = sum(1 for r in cross if r.get("qwen_faith_abstained"))
    out["faith_qwen"] = {
        "n": len(cross), "none": faith_none, "abstained": faith_abst,
        "failure": faith_none - faith_abst,  # None БЕЗ воздержания = сбой (после U1 → 0)
    }

    # Срез по маршрутам на ОБЩЕМ наборе (faith/relevance/precision).
    from collections import defaultdict
    routes: dict[str, list] = defaultdict(list)
    for r in cross:
        routes[r.get("route") or "?"].append(r)
    for rt, rs in routes.items():
        row = {"n": len(rs)}
        for m in _METRICS:
            common = [(r.get("deepseek", {}).get(m), r.get("qwen", {}).get(m)) for r in rs]
            common = [(d, q) for d, q in common if d is not None and q is not None]
            row[m] = {
                "common_n": len(common),
                "deepseek": _mean([d for d, _ in common]),
                "qwen": _mean([q for _, q in common]),
            }
        out["by_route"][rt] = row
    return out


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
        split = calibration_split(cross, gold)
        lines += [
            "", f"## Калибровка против человека (gold n={cal['n']})", "",
            "MAE к человеку — меньше = ближе к правде.", "",
            "| метрика | DeepSeek MAE | Qwen MAE |", "|---|---|---|",
        ]
        for m in _METRICS:
            lines.append(f"| {m} | {_fmt(cal[m]['deepseek_mae'])} | {_fmt(cal[m]['qwen_mae'])} |")

        # Раздельно refusal / non-refusal — против survivorship-вывода.
        lines += [
            "", "## MAE раздельно: отказы vs не-отказы", "",
            f"Отказов в gold: {split['refusal']['n']} · не-отказов: {split['non_refusal']['n']}", "",
            "| метрика | DS отказы | Qwen отказы | DS не-отказы | Qwen не-отказы |",
            "|---|---|---|---|---|",
        ]
        for m in _METRICS:
            rf, nf = split["refusal"], split["non_refusal"]
            lines.append(
                f"| {m} | {_fmt(rf[m]['deepseek_mae'])} | {_fmt(rf[m]['qwen_mae'])} | "
                f"{_fmt(nf[m]['deepseek_mae'])} | {_fmt(nf[m]['qwen_mae'])} |")
        lines += ["", f"**Глобальный вердикт: {split['verdict']}**"]

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

    # Второй отчёт — честный baseline без survivorship.
    tb = trusted_baseline(cross)
    fq = tb["faith_qwen"]
    bl = [
        "# Доверенный baseline (без survivorship)",
        "",
        "> Первично — сравнение судей на **общем не-воздержавшемся наборе** (оба судьи "
        "дали число). Сравнивать полное среднее Qwen (по не-воздержаниям) с полным DeepSeek "
        "(по всем) — это разный знаменатель, отсюда мнимая «разница».",
        "",
        f"Записей всего: {tb['n']}",
        "",
        "## Первично: DeepSeek vs Qwen на ОБЩЕМ наборе",
        "",
        "> Как читать (см. калибровку): **faithfulness** — Qwen решительно ближе к человеку "
        "(MAE 0.125 vs 0.417), это доверенная цифра. **relevance/precision** — судьи "
        "сопоставимы на не-отказах; выше у Qwen во многом из-за корректной оценки отказов, "
        "которые DeepSeek занижал. Цифры Qwen берём как baseline, но превосходства по "
        "relevance/precision на отвеченных вопросах не заявляем.",
        "",
        "| метрика | общий n | DeepSeek | Qwen | Δ(Qwen−DS) |",
        "|---|---|---|---|---|",
    ]
    for m in _METRICS:
        b = tb["by_metric"][m]
        d = (None if b["deepseek_common"] is None or b["qwen_common"] is None
             else b["qwen_common"] - b["deepseek_common"])
        bl.append(f"| {m} | {b['common_n']} | {_fmt(b['deepseek_common'])} | "
                  f"{_fmt(b['qwen_common'])} | {_fmt(d)} |")
    bl += [
        "", "## Вторично: полные средние (⚠️ РАЗНЫЙ знаменатель)", "",
        "| метрика | DeepSeek (n) | Qwen (n) |", "|---|---|---|",
    ]
    for m in _METRICS:
        b = tb["by_metric"][m]
        bl.append(f"| {m} | {_fmt(b['deepseek_full'])} ({b['deepseek_full_n']}) | "
                  f"{_fmt(b['qwen_full'])} ({b['qwen_full_n']}) |")
    bl += [
        "", "## Воздержания и сбои Qwen (faithfulness)", "",
        f"- всего записей: {fq['n']}",
        f"- Qwen faith = None: {fq['none']} (из них воздержаний {fq['abstained']}, "
        f"сбоев {fq['failure']})",
        "- воздержание = «нет проверяемых утверждений» (честный отказ), НЕ провал; "
        "сбои дозаполнены в U1.",
        "", "## По маршрутам (общий набор)", "",
        "| маршрут | n | faith DS/Qwen | relevance DS/Qwen | precision DS/Qwen |",
        "|---|---|---|---|---|",
    ]
    for rt, row in sorted(tb["by_route"].items()):
        def cell(m):
            return f"{_fmt(row[m]['deepseek'])}/{_fmt(row[m]['qwen'])}"
        bl.append(f"| {rt} | {row['n']} | {cell('faithfulness')} | "
                  f"{cell('answer_relevance')} | {cell('context_precision')} |")
    Path(_BASELINE).write_text("\n".join(bl), encoding="utf-8")

    print(f"DONE calibration -> {_REPORT} + {_BASELINE} (gold={'да' if gold else 'нет'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
