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
