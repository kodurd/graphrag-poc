#!/usr/bin/env bash
# Бэкап графа Neo4j: тар bind-mount тома neo4j_data (Community + Docker).
# Зачем: граф однажды потерялся (пустая БД после destructive-операции), а пересборка
# с нуля = ~1.5ч локальных эмбеддингов. Бэкап восстанавливается за секунды.
# Community edition dump требует остановки БД, поэтому просто останавливаем контейнер
# и тарим том. Использование: scripts/neo4j_backup.sh [метка]
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=graphrag-neo4j
DATA_DIR=neo4j_data
BACKUP_DIR=backups
LABEL="${1:-$(printf 'manual')}"
STAMP=$(printf '%s' "$(date +%Y%m%d-%H%M%S 2>/dev/null || echo nodate)")
OUT="$BACKUP_DIR/neo4j_${LABEL}_${STAMP}.tar.gz"

mkdir -p "$BACKUP_DIR"
echo "[backup] останавливаю $CONTAINER (кратко, БД должна быть offline для консистентного тара)"
docker stop "$CONTAINER" >/dev/null
echo "[backup] тарю $DATA_DIR -> $OUT"
tar czf "$OUT" "$DATA_DIR"
echo "[backup] запускаю $CONTAINER"
docker start "$CONTAINER" >/dev/null
echo "[backup] готово: $OUT ($(du -h "$OUT" | cut -f1))"
echo "[backup] восстановление: scripts/neo4j_restore.sh $OUT"
