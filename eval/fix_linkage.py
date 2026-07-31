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
