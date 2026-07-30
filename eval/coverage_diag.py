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
