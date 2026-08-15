#!/usr/bin/env bash
# Честный A/B neighbor-expansion на ОДНОМ графе (без пересборки).
# before: kip_neighbors=0 (прод), after: kip_neighbors=1. reserve=2 в обоих.
# Снимки читают граф (не вайпают) -> прерывание безопасно, теряется только прогресс.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export KIP_RESERVE=2

OUT=eval/trial/neighbor_ab
mkdir -p "$OUT"
LOG="$OUT/run.log"
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "START neighbor A/B (reserve=2, neighbors 0 vs 1)"

say "1/5 snapshot BEFORE (neighbors=0)"
KIP_NEIGHBORS=0 QS_OUT="$OUT/snap_n0.json" uv run --extra ml python -m eval.quality_snapshot >>"$LOG" 2>&1 || { say "FAIL snap_n0"; exit 1; }

say "2/5 cross-judge BEFORE"
CJ_IN="$OUT/snap_n0.json" CJ_OUT="$OUT/cj_n0.json" uv run python -m eval.cross_judge >>"$LOG" 2>&1 || { say "FAIL cj_n0"; exit 1; }

say "3/5 snapshot AFTER (neighbors=1)"
KIP_NEIGHBORS=1 QS_OUT="$OUT/snap_n1.json" uv run --extra ml python -m eval.quality_snapshot >>"$LOG" 2>&1 || { say "FAIL snap_n1"; exit 1; }

say "4/5 cross-judge AFTER"
CJ_IN="$OUT/snap_n1.json" CJ_OUT="$OUT/cj_n1.json" uv run python -m eval.cross_judge >>"$LOG" 2>&1 || { say "FAIL cj_n1"; exit 1; }

say "5/5 kip_delta"
KIP_BEFORE="$OUT/snap_n0.json" KIP_AFTER="$OUT/snap_n1.json" \
KIP_CJ_BEFORE="$OUT/cj_n0.json" KIP_CJ_AFTER="$OUT/cj_n1.json" \
  uv run python -m eval.kip_delta >>"$LOG" 2>&1 || { say "FAIL kip_delta"; exit 1; }

say "DONE neighbor A/B"
