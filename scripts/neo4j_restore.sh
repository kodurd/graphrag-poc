#!/usr/bin/env bash
# Восстановление графа Neo4j из тар-бэкапа (см. neo4j_backup.sh).
# Использование: scripts/neo4j_restore.sh backups/neo4j_XXXX.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=graphrag-neo4j
DATA_DIR=neo4j_data
ARCHIVE="${1:?укажи путь к .tar.gz бэкапу}"
[ -f "$ARCHIVE" ] || { echo "нет файла: $ARCHIVE" >&2; exit 1; }

echo "[restore] останавливаю $CONTAINER"
docker stop "$CONTAINER" >/dev/null
echo "[restore] бэкаплю текущий $DATA_DIR -> ${DATA_DIR}.prev (на всякий)"
rm -rf "${DATA_DIR}.prev"; [ -d "$DATA_DIR" ] && mv "$DATA_DIR" "${DATA_DIR}.prev" || true
echo "[restore] распаковываю $ARCHIVE"
tar xzf "$ARCHIVE"
echo "[restore] запускаю $CONTAINER"
docker start "$CONTAINER" >/dev/null
echo "[restore] готово. Проверь: MATCH (n) RETURN count(n)"
