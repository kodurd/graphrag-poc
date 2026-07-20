"""Слой 5 — инкрементальные обновления: пересчёт только затронутого."""

from graphrag.incremental.sync import (
    IncrementalSync,
    build_manifest,
    diff_manifests,
    record_key,
)

__all__ = ["IncrementalSync", "build_manifest", "diff_manifests", "record_key"]
