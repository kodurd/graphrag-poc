"""Выбор пути вывода замера: env перекрывает дефолт, иначе backward-compat.

Изоляция BEFORE/AFTER (U3): резолверы путей — чистые (без диска), чтобы прогоны
писали в разные файлы на одном билде, не перетирая друг друга. Тестируем именно
резолвер, не запись на диск.
"""

from __future__ import annotations

from pathlib import Path

from eval.cross_judge import _resolve_in, _resolve_out as _cj_resolve_out
from eval.quality_snapshot import _RESULTS, _resolve_out


def test_qs_out_env_overrides_default(monkeypatch):
    monkeypatch.setenv("QS_OUT", "eval/trial/quality_snapshot_results_before.json")
    assert _resolve_out(_RESULTS) == Path("eval/trial/quality_snapshot_results_before.json")


def test_qs_out_unset_keeps_default(monkeypatch):
    """Backward-compat: без env резолвер возвращает текущий дефолтный путь."""
    monkeypatch.delenv("QS_OUT", raising=False)
    assert _resolve_out(_RESULTS) == _RESULTS


def test_qs_out_nonexistent_dir_returned_as_is(monkeypatch):
    """Путь в несуществующем каталоге возвращается как есть — создание/ошибка на
    писателе. Не должно молча откатываться к дефолту."""
    target = "eval/trial/does_not_exist/snapshot_after.json"
    monkeypatch.setenv("QS_OUT", target)
    resolved = _resolve_out(_RESULTS)
    assert resolved == Path(target)
    assert resolved != _RESULTS


def test_cj_in_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CJ_IN", "eval/trial/quality_snapshot_results_after.json")
    assert _resolve_in("eval/trial/quality_snapshot_results.json") == Path(
        "eval/trial/quality_snapshot_results_after.json"
    )


def test_cj_in_unset_keeps_default(monkeypatch):
    monkeypatch.delenv("CJ_IN", raising=False)
    default = "eval/trial/quality_snapshot_results.json"
    assert _resolve_in(default) == Path(default)


def test_cj_out_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CJ_OUT", "eval/trial/cross_judge_results_after.json")
    assert _cj_resolve_out("eval/trial/cross_judge_results.json") == Path(
        "eval/trial/cross_judge_results_after.json"
    )


def test_cj_out_unset_keeps_default(monkeypatch):
    monkeypatch.delenv("CJ_OUT", raising=False)
    default = "eval/trial/cross_judge_results.json"
    assert _cj_resolve_out(default) == Path(default)


def test_cj_out_nonexistent_dir_returned_as_is(monkeypatch):
    """Несуществующий каталог не подменяется дефолтом — writer сам решает."""
    target = "eval/trial/nope/cross_after.json"
    monkeypatch.setenv("CJ_OUT", target)
    resolved = _cj_resolve_out("eval/trial/cross_judge_results.json")
    assert resolved == Path(target)
    assert resolved != Path("eval/trial/cross_judge_results.json")
