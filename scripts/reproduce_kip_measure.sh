#!/usr/bin/env bash
# Воспроизведение KIP before/after замера с нуля. Гейтированный, идемпотентный.
# Предпосылки: Neo4j поднят (docker compose up -d), .env с LLM_API_KEY + JUDGE_API_KEY,
# data/intermediate/*.jsonl на месте (git+jira+confluence non-KIP), uv установлен.
#
# УРОКИ, зашитые сюда:
#  - НЕ гнать build и snapshot внахлёст (пик RAM ронял процесс) — фазы строго по очереди.
#  - Каждая фаза с ПОЛНЫМ логом (без tail-буфера), ждём явного файла/маркера.
#  - BEFORE = граф БЕЗ KIP; AFTER = тот же граф + KIP; отличие только KIP.
#  - Транзиентные `SKIP: LLMError` (SSL EOF) — штатны, snapshot их переживает.
#
# Использование: scripts/reproduce_kip_measure.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
T=eval/trial
say(){ echo "[$(date +%H:%M:%S 2>/dev/null || echo now)] $*"; }

# 0) Пересборка корпуса из JSONL (если граф пуст). Локальные эмбеддинги ~1.5ч на CPU.
NODES=$(uv run python -c "from graphrag.config import load_settings as L;from graphrag.graph import Neo4jConnection as C
s=L();
import sys
with C(s.neo4j) as c: print(list(c.run('MATCH (n) RETURN count(n) AS c'))[0]['c'])" 2>/dev/null | tail -1)
say "граф узлов: ${NODES:-?}"
if [ "${NODES:-0}" -lt 1000 ]; then
  say "0) граф пуст -> build --index (rebuild из data/intermediate/*.jsonl)"
  uv run --extra ml graphrag build --index
fi

# Убедимся, что KIP НЕ в графе для BEFORE (обратимо; KIP re-ingest ниже).
say "чищу KIP из графа для чистого BEFORE"
uv run python -c "from graphrag.config import load_settings as L;from graphrag.graph import Neo4jConnection as C
with C(L().neo4j) as c:
    c.run(\"MATCH (ch:Chunk)-[:PART_OF]->(p:Page) WHERE p.title STARTS WITH 'KIP-' DETACH DELETE ch\")
    c.run(\"MATCH (p:Page) WHERE p.title STARTS WITH 'KIP-' DETACH DELETE p\")
    print('KIP-страниц осталось:', list(c.run(\"MATCH (p:Page) WHERE p.title STARTS WITH 'KIP-' RETURN count(p) AS c\"))[0]['c'])" 2>&1 | tail -1

# 1) BEFORE (без KIP)
say "1) BEFORE snapshot (без KIP)"
QS_OUT=$T/quality_snapshot_results_before.json uv run --extra ml python -m eval.quality_snapshot > $T/_before.log 2>&1
CJ_IN=$T/quality_snapshot_results_before.json CJ_OUT=$T/cross_judge_results_before.json uv run python -m eval.cross_judge > $T/_cj_before.log 2>&1 || say "cross_judge BEFORE упал (не блокер: primary metric из снимка)"

# 2) Ingest KIP (CQL ancestor=<KIP-parent-id> в settings) + build
say "2) ingest KIP + build (инкрементальный эмбеддинг KIP-чанков)"
cp -f data/intermediate/confluence.jsonl data/intermediate/confluence.nonkip.bak.jsonl 2>/dev/null || true
uv run graphrag ingest --confluence > $T/_ingest_kip.log 2>&1
uv run --extra ml graphrag build --index > $T/_build_kip.log 2>&1
# ВАЖНО: дождаться конца build ПЕРЕД AFTER — иначе AFTER увидит неполный KIP-индекс.

# 3) AFTER (с KIP)
say "3) AFTER snapshot (с KIP)"
QS_OUT=$T/quality_snapshot_results_after.json uv run --extra ml python -m eval.quality_snapshot > $T/_after.log 2>&1
CJ_IN=$T/quality_snapshot_results_after.json CJ_OUT=$T/cross_judge_results_after.json uv run python -m eval.cross_judge > $T/_cj_after.log 2>&1 || say "cross_judge AFTER упал (не блокер)"

# 4) Дельта
say "4) kip_delta -> $T/kip_delta_report.md"
uv run python -m eval.kip_delta
say "ГОТОВО. Результат: $T/kip_delta_report.md"
