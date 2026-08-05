"""Дельта-анализ ДО/ПОСЛЕ ingest KIP — честный вердикт по гипотезе жанра.

Гипотеза (см. план 2026-08-05-001): добавление how-to-жанрового источника (KIP)
в корпус снимает вынужденные отказы. Наивная метрика «доля ответов там, где KIP
дошли в top-k» условлена на самом тестируемом механизме ретрива и завышает эффект.
Поэтому вердикт строится на СТОРОЖАХ (из ревью плана):

- Первичный лифт (R6) — answer-rate на ВСЕХ ранее-безответных (полный знаменатель),
  а «KIP-в-top-k» — диагностический срез рядом, со своим размером.
- R7 reachability-гейт: если KIP реально не дошли в top-k — нулевой лифт есть сбой
  ретрива, НЕ отрицание жанра (retrieval failure, not genre negative).
- R8 шумовой пол: лифт валиден, только если превышает флип-рейт `is_refusal` на
  неизменном корпусе (передаётся извне — из повторного BEFORE, здесь не считается).
- R9 displacement: Qwen-faithfulness на ранее-отвеченных не просела.

Все аналитические функции ЧИСТЫЕ (берут списки dict, без I/O и без LLM); склейка
снимков и запись отчёта — только в main().

Запуск:
    uv run python -m eval.kip_delta [before.json] [after.json]
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from eval.human_gold import is_refusal

_BEFORE = "eval/trial/quality_snapshot_results_before.json"
_AFTER = "eval/trial/quality_snapshot_results_after.json"
_CJ_BEFORE = "eval/trial/cross_judge_results_before.json"
_CJ_AFTER = "eval/trial/cross_judge_results_after.json"
_REPORT = "eval/trial/kip_delta_report.md"

# Префикс id чанка Confluence/KIP-страницы (в отличие от chunk:task:/chunk:commit:).
_KIP_PREFIX = "chunk:page:"


def _key(r: dict) -> tuple:
    return (r.get("source_id"), r.get("question"))


def join_before_after(before: list[dict], after: list[dict]) -> list[tuple]:
    """Пары (before_rec, after_rec) по (source_id, question) — только пересечение.

    Замороженный набор вопросов (R4) => ключи совпадают. Дедуп по ключу с обеих
    сторон (последняя запись выигрывает), иначе дубликат ключа удвоил бы счётчики.
    Несопоставленные записи учитываются отдельно (`unmatched_unanswered`), а не
    молча теряются — иначе усадка знаменателя завысила бы лифт.
    """
    before_by_key = {_key(r): r for r in before}
    after_by_key = {_key(r): r for r in after}
    return [(b, after_by_key[k]) for k, b in before_by_key.items()
            if k in after_by_key]


def unmatched_unanswered(before: list[dict], after: list[dict]) -> int:
    """Сколько ранее-безответных (refusal в BEFORE) НЕ нашли пары в AFTER.

    Телеметрия честности знаменателя: >0 означает, что набор ПОСЛЕ неполон
    (прерванный/сбойный прогон), и первичный лифт считается на усечённом множестве —
    об этом надо предупредить, а не молча завышать rate.
    """
    after_keys = {_key(r) for r in after}
    seen: set = set()
    n = 0
    for b in before:
        k = _key(b)
        if k in seen:
            continue
        seen.add(k)
        if is_refusal(b) and k not in after_keys:
            n += 1
    return n


def _kip_ids(rec: dict) -> list[str]:
    """context_ids записи, указывающие на KIP/Confluence-страницу (chunk:page:...)."""
    return [c for c in (rec.get("context_ids") or [])
            if isinstance(c, str) and c.startswith(_KIP_PREFIX)]


def kip_in_topk(after_rec: dict) -> bool:
    """Достиг ли хоть один KIP/Confluence-чанк top-k этого ответа."""
    return bool(_kip_ids(after_rec))


def classify_transition(before_rec: dict, after_rec: dict) -> str:
    """Переход отказ/ответ между BEFORE и AFTER по `is_refusal` на каждом."""
    br, ar = is_refusal(before_rec), is_refusal(after_rec)
    if br and not ar:
        return "refusal_to_answered"
    if not br and ar:
        return "answered_to_refusal"
    if not br and not ar:
        return "stayed_answered"
    return "stayed_refusal"


def primary_lift(before: list[dict], after: list[dict]) -> dict:
    """Первичный лифт (R6): answer-rate на ВСЕХ ранее-безответных.

    Знаменатель = все, кто был refusal в BEFORE (там answer-rate 0 по построению);
    числитель = сколько из них перестали быть refusal в AFTER. rate = лифт.
    """
    pairs = join_before_after(before, after)
    prev = [(b, a) for b, a in pairs if is_refusal(b)]
    denom = len(prev)
    num = sum(1 for _, a in prev if not is_refusal(a))
    return {"numerator": num, "denominator": denom,
            "rate": (num / denom if denom else None)}


def diagnostic_subset(before: list[dict], after: list[dict]) -> dict:
    """Диагностический срез: среди ранее-безответных — где KIP дошли в top-k.

    Отдельно от первичного лифта (у него полный знаменатель). Возвращает размер
    среза, сколько из него стало отвечено, и число ранее-безответных, где НИ ОДИН
    KIP не дошёл в top-k (кандидаты на R7 «сбой ретрива»).
    """
    pairs = join_before_after(before, after)
    prev = [(b, a) for b, a in pairs if is_refusal(b)]
    kip = [(b, a) for b, a in prev if kip_in_topk(a)]
    no_kip = [(b, a) for b, a in prev if not kip_in_topk(a)]
    became_answered = sum(1 for _, a in kip if not is_refusal(a))
    return {"previously_unanswered": len(prev), "size": len(kip),
            "became_answered": became_answered, "no_kip_count": len(no_kip)}


def reachability_gate(after_over_unanswered: list[dict], *, floor: int = 0) -> str:
    """R7-гейт: дошли ли KIP реально в top-k ранее-безответных.

    Вход — AFTER-записи по ранее-безответному множеству. Если KIP-в-top-k срез пуст
    (≤ floor), нулевой/малый лифт означает СБОЙ РЕТРИВА, а не отрицание жанра —
    интерпретируемый исход, а не молчаливый ноль.
    """
    n = len(after_over_unanswered)
    kip = sum(1 for a in after_over_unanswered if kip_in_topk(a))
    if kip <= floor:
        return ("R7-гейт: retrieval failure, not genre negative — "
                f"KIP дошли в top-k лишь в {kip}/{n} ранее-безответных (floor={floor}); "
                "нулевой лифт = сбой ретрива, НЕ отрицание жанра.")
    return (f"R7-гейт: reachable — KIP дошли в top-k в {kip}/{n} ранее-безответных; "
            "лифт интерпретируем как эффект жанра.")


def noise_floor_ok(observed_lift_count: int, flip_rate_count: int) -> bool:
    """R8: лифт валиден, только если СТРОГО превышает флип-рейт `is_refusal`.

    flip_rate_count — вход (число флипов на неизменном корпусе из повторного BEFORE),
    здесь НЕ вычисляется. Равенство = в пределах шума => False.
    """
    return observed_lift_count > flip_rate_count


def _qwen_faith(rec: dict):
    """Аксессор Qwen-faithfulness после attach_qwen_faith (или inline, если есть)."""
    return rec.get("qwen_faithfulness")


def displacement_check(before: list[dict], after: list[dict], *,
                       faith_before_key, faith_after_key) -> list[dict]:
    """R9: среди stayed_answered — вопросы, где Qwen-faithfulness ПРОСЕЛА (регресс).

    faith_before_key/faith_after_key — аксессоры (callable по записи), поэтому работает
    и когда faith инлайн в снимке, и когда приджойнена из cross_judge. Пары с None
    хотя бы с одной стороны пропускаем (нечего сравнивать).
    """
    out: list[dict] = []
    for b, a in join_before_after(before, after):
        if is_refusal(b) or is_refusal(a):
            continue  # только те, кто был и остался отвечён
        fb, fa = faith_before_key(b), faith_after_key(a)
        if fb is None or fa is None:
            continue
        if fa < fb:
            out.append({"source_id": b.get("source_id"), "question": b.get("question"),
                        "faith_before": fb, "faith_after": fa, "drop": fb - fa})
    return out


def attach_qwen_faith(records: list[dict], cross_records: list[dict]) -> list[dict]:
    """Чистый merge: приджойнить Qwen-faithfulness из cross_judge по (source_id, question).

    Возвращает НОВЫЙ список копий (вход не мутируется); к каждой добавляет
    `qwen_faithfulness` и `qwen_faith_abstained`, если нашлась пара в cross.
    """
    by_key = {_key(c): c for c in (cross_records or [])}
    out: list[dict] = []
    for r in records:
        merged = dict(r)
        c = by_key.get(_key(r))
        if c is not None:
            merged["qwen_faithfulness"] = (c.get("qwen") or {}).get("faithfulness")
            merged["qwen_faith_abstained"] = c.get("qwen_faith_abstained")
        out.append(merged)
    return out


def _examples(before: list[dict], after: list[dict], *, limit: int = 5) -> list[dict]:
    """Строки «был отказ -> теперь ответ со ссылкой на KIP» (для отчёта)."""
    rows: list[dict] = []
    for b, a in join_before_after(before, after):
        if not is_refusal(b) or is_refusal(a) or not kip_in_topk(a):
            continue
        kids = _kip_ids(a)
        rows.append({"source_id": a.get("source_id"), "question": a.get("question"),
                     "kip_id": kids[0] if kids else None,
                     "answer": (a.get("answer") or "")[:200]})
        if len(rows) >= limit:
            break
    return rows


def report(before: list[dict], after: list[dict], *, flip_rate_count: int = 0,
           faith_before_key=_qwen_faith, faith_after_key=_qwen_faith,
           r7_floor: int = 0, examples_limit: int = 5) -> dict:
    """Собирает весь вердикт: первичный лифт, диагностический срез, R7/R8/R9, примеры.

    Чистая — принимает уже склеенные списки (faith при необходимости приджойнена
    через attach_qwen_faith до вызова).
    """
    pairs = join_before_after(before, after)
    prev_unans_after = [a for b, a in pairs if is_refusal(b)]

    lift = primary_lift(before, after)
    diag = diagnostic_subset(before, after)
    r8_ok = noise_floor_ok(lift["numerator"], flip_rate_count)
    disp = displacement_check(before, after, faith_before_key=faith_before_key,
                              faith_after_key=faith_after_key)
    transitions = Counter(classify_transition(b, a) for b, a in pairs)

    return {
        "n_joined": len(pairs),
        "unmatched_unanswered": unmatched_unanswered(before, after),
        "transitions": dict(transitions),
        "primary_lift": lift,
        "diagnostic": diag,
        "r7": reachability_gate(prev_unans_after, floor=r7_floor),
        "r8": {"ok": r8_ok, "observed": lift["numerator"], "flip_rate": flip_rate_count,
               "verdict": ("R8: лифт выше шумового пола — валиден"
                           if r8_ok else
                           "R8: лифт в пределах шума (<= флип-рейт) — НЕ валиден")},
        "r9": {"regressions": disp,
               "verdict": (f"R9: displacement — просела faithfulness у {len(disp)} "
                           "ранее-отвеченных" if disp else
                           "R9: displacement не обнаружен (faithfulness не просела)")},
        "examples": _examples(before, after, limit=examples_limit),
    }


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_report(rep: dict) -> str:
    """Markdown-отчёт из структуры report()."""
    pl, dg = rep["primary_lift"], rep["diagnostic"]
    lines = [
        "# KIP дельта-анализ ДО/ПОСЛЕ — вердикт по гипотезе жанра",
        "",
        "> Первичный критерий — лифт на **всех ранее-безответных** (полный знаменатель). "
        "«KIP-в-top-k» — диагностика, НЕ со-равный критерий. Вердикт гейтирован сторожами "
        "R7 (reachability), R8 (шумовой пол), R9 (displacement).",
        "",
        f"Сопоставлено пар (source_id, question): {rep['n_joined']}",
        "",
        "## Переходы",
        "",
        "| переход | n |", "|---|---|",
    ]
    for k in ("refusal_to_answered", "answered_to_refusal",
              "stayed_answered", "stayed_refusal"):
        lines.append(f"| {k} | {rep['transitions'].get(k, 0)} |")

    lines += [
        "", "## Первичный лифт (все ранее-безответные)", "",
        f"- ранее-безответных (знаменатель): **{pl['denominator']}**",
        f"- стали отвечены (числитель): **{pl['numerator']}**",
        f"- лифт answer-rate: **{_fmt(pl['rate'])}**",
    ]
    _unm = rep.get("unmatched_unanswered", 0)
    if _unm:
        lines.append(
            f"- ⚠️ **{_unm} ранее-безответных не нашли пары в AFTER** — знаменатель "
            "усечён, лифт может быть завышен; пере-снять AFTER на полном наборе.")
    lines += [
        "",
        "## Диагностический срез: KIP-в-top-k (НЕ первичный критерий)", "",
        f"- размер среза (KIP дошли в top-k): {dg['size']}",
        f"- из них стали отвечены: {dg['became_answered']}",
        f"- ранее-безответных БЕЗ KIP в top-k: {dg['no_kip_count']} "
        f"(из {dg['previously_unanswered']})",
        "",
        "## Сторожа интерпретации", "",
        f"- **{rep['r7']}**",
        f"- **{rep['r8']['verdict']}** "
        f"(лифт {rep['r8']['observed']} vs флип-рейт {rep['r8']['flip_rate']})",
        f"- **{rep['r9']['verdict']}**",
    ]
    if rep["r9"]["regressions"]:
        lines += ["", "| source_id | вопрос | faith ДО | faith ПОСЛЕ | Δ |",
                  "|---|---|---|---|---|"]
        for d in rep["r9"]["regressions"]:
            lines.append(f"| {d['source_id']} | {(d['question'] or '')[:60]} | "
                         f"{_fmt(d['faith_before'])} | {_fmt(d['faith_after'])} | "
                         f"{_fmt(d['drop'])} |")

    lines += ["", "## Примеры: был отказ -> ответ со ссылкой на KIP", ""]
    if rep["examples"]:
        for i, e in enumerate(rep["examples"], 1):
            lines += [
                f"### {i}. {e['source_id']}  (KIP: {e['kip_id']})",
                f"**Вопрос:** {e['question']}",
                f"**Ответ:** {e['answer']}",
                "",
            ]
    else:
        lines += ["_нет примеров refusal->answered с KIP в top-k_", ""]
    return "\n".join(lines)


def _load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))["records"]


def main() -> int:
    import sys

    before_p = Path(sys.argv[1] if len(sys.argv) > 1
                    else os.environ.get("KIP_BEFORE", _BEFORE))
    after_p = Path(sys.argv[2] if len(sys.argv) > 2
                   else os.environ.get("KIP_AFTER", _AFTER))
    if not before_p.exists() or not after_p.exists():
        print(f"kip-delta: net snapshotov ({before_p} / {after_p}) -- "
              "snachala progon BEFORE i AFTER (quality_snapshot)")
        return 1

    before = _load_records(before_p)
    after = _load_records(after_p)

    # Опциональный джойн Qwen-faith из cross_judge (для R9 displacement).
    cj_before_p = Path(os.environ.get("KIP_CJ_BEFORE", _CJ_BEFORE))
    cj_after_p = Path(os.environ.get("KIP_CJ_AFTER", _CJ_AFTER))
    if cj_before_p.exists():
        before = attach_qwen_faith(before, _load_records(cj_before_p))
    if cj_after_p.exists():
        after = attach_qwen_faith(after, _load_records(cj_after_p))

    flip_rate_count = int(os.environ.get("KIP_FLIP_COUNT", "0"))
    r7_floor = int(os.environ.get("KIP_R7_FLOOR", "0"))

    rep = report(before, after, flip_rate_count=flip_rate_count, r7_floor=r7_floor)
    Path(_REPORT).write_text(render_report(rep), encoding="utf-8")

    pl = rep["primary_lift"]
    print(f"DONE kip-delta: lift {pl['numerator']}/{pl['denominator']}, "
          f"kip-subset {rep['diagnostic']['size']}, r8_ok={rep['r8']['ok']}, "
          f"displacement={len(rep['r9']['regressions'])} -> {_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
