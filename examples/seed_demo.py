"""Засеять демо-граф (без реального корпуса) для ручной проверки сценариев
`graphrag log-impact`, `graphrag ask` и генерации golden set.

Ядро clients/connect/streams сохранено, чтобы примеры из README работали;
streams намеренно оставлен «листом» (никто от него не зависит). Вокруг ядра —
расширенный срез Kafka: модули с зависимостями, тикеты со связями
DUPLICATES/FIXED_BY (из них строится golden set) и вики-страницы (KIP).

Запуск: uv run python examples/seed_demo.py
"""

from __future__ import annotations

from graphrag.config import load_settings
from graphrag.graph import Neo4jConnection
from graphrag.graph.schema import apply_schema
from graphrag.graph.skeleton import load_records
from graphrag.intermediate import edge, node

# --- Модули и зависимости (DEPENDS_ON: зависящий -> зависимость) ---------------
# clients — базовая клиентская библиотека, от неё зависит почти всё.
# streams оставлен листом (нет входящих DEPENDS_ON) — для негативного кейса.
_MODULES = [
    "clients", "common", "connect", "streams", "core",
    "coordinator", "raft", "storage", "tools", "metadata",
]
_DEPENDS_ON = [
    ("connect", "clients"),
    ("streams", "clients"),
    ("tools", "clients"),
    ("core", "clients"),
    ("clients", "common"),
    ("core", "common"),
    ("connect", "core"),
    ("coordinator", "core"),
    ("raft", "core"),
    ("storage", "core"),
    ("metadata", "raft"),
]

# --- Тикеты (JIRA-подобные) ----------------------------------------------------
# (key, summary, status, [модули, которые упоминает], владелец)
_TASKS = [
    ("KAFKA-101", "NetworkClient reconnect loop under broker outage", "Resolved",
     ["clients"], "ann"),
    ("KAFKA-102", "Producer retries exhausted on broker restart", "Resolved",
     ["clients"], "bob"),
    ("KAFKA-110", "WorkerSourceTask fails when broker unavailable", "Resolved",
     ["connect", "clients"], "ann"),
    ("KAFKA-111", "Sink connector stuck after consumer group rebalance", "In Progress",
     ["connect"], "carol"),
    ("KAFKA-112", "Connector emits duplicate records after retry", "Resolved",
     ["connect"], "bob"),
    ("KAFKA-120", "StreamThread dies on task migration", "Resolved",
     ["streams"], "dmitry"),
    ("KAFKA-121", "Streams state store restore is slow after rebalance", "Open",
     ["streams"], "dmitry"),
    ("KAFKA-130", "Controller failover leaves stale metadata", "Resolved",
     ["core", "metadata"], "carol"),
    ("KAFKA-131", "Group coordinator NPE on member session timeout", "Resolved",
     ["coordinator"], "bob"),
    ("KAFKA-140", "Raft leader election storm under network partition", "Resolved",
     ["raft"], "ann"),
    ("KAFKA-150", "Log segment corruption on unclean shutdown", "Resolved",
     ["storage"], "dmitry"),
    ("KAFKA-160", "Console consumer hangs on empty topic", "Closed",
     ["tools"], "carol"),
]

# Связи между тикетами: дубликаты и «починен тикетом» (из них строится golden).
_DUPLICATES = [
    ("KAFKA-102", "KAFKA-101"),  # тот же reconnect-баг с другой стороны
    ("KAFKA-112", "KAFKA-110"),  # дубли записей — следствие того же падения
    ("KAFKA-121", "KAFKA-120"),  # обе про restore/миграцию state store
]
_FIXED_BY = [
    ("KAFKA-101", "KAFKA-102"),
    ("KAFKA-110", "KAFKA-112"),
]

# --- Люди ----------------------------------------------------------------------
_PEOPLE = {
    "ann": "Ann Dev",
    "bob": "Bob Ops",
    "carol": "Carol Lead",
    "dmitry": "Dmitry Streams",
}

# --- Вики-страницы (KIP), упоминают тикеты -------------------------------------
_PAGES = [
    ("page:kip-networking", "KIP-Networking: client retry semantics",
     "https://cwiki.apache.org/confluence/display/KAFKA/KIP-Networking", "KAFKA-101"),
    ("page:kip-connect", "KIP-Connect: rebalance and exactly-once",
     "https://cwiki.apache.org/confluence/display/KAFKA/KIP-Connect", "KAFKA-111"),
    ("page:kip-streams", "KIP-Streams: state store restore",
     "https://cwiki.apache.org/confluence/display/KAFKA/KIP-Streams", "KAFKA-120"),
    ("page:kip-raft", "KIP-Raft: leader election under partitions",
     "https://cwiki.apache.org/confluence/display/KAFKA/KIP-Raft", "KAFKA-140"),
]


def _build_records() -> list[dict]:
    records: list[dict] = []

    for name in _MODULES:
        records.append(node("Module", f"module:{name}", {"name": name}))
    for dependent, dependency in _DEPENDS_ON:
        records.append(edge("DEPENDS_ON", f"module:{dependent}", f"module:{dependency}"))

    for pid, name in _PEOPLE.items():
        records.append(node("Person", f"person:{pid}", {"name": name}))

    for key, summary, status, modules, owner in _TASKS:
        tid = f"task:{key}"
        records.append(node("Task", tid, {
            "key": key, "summary": summary, "status": status,
            "uri": f"https://issues.apache.org/jira/browse/{key}",
        }))
        for m in modules:
            records.append(edge("MENTIONS", tid, f"module:{m}"))
        records.append(edge("ASSIGNED_TO", tid, f"person:{owner}"))

    for a, b in _DUPLICATES:
        records.append(edge("DUPLICATES", f"task:{a}", f"task:{b}"))
    for a, b in _FIXED_BY:
        records.append(edge("FIXED_BY", f"task:{a}", f"task:{b}"))

    for pid, title, uri, task_key in _PAGES:
        records.append(node("Page", pid, {"title": title, "uri": uri}))
        records.append(edge("MENTIONS", pid, f"task:{task_key}"))

    return records


DEMO = _build_records()


def main() -> None:
    settings = load_settings()
    with Neo4jConnection(settings.neo4j) as conn:
        apply_schema(conn)
        stats = load_records(conn, DEMO)
        print(f"seeded: {stats}")


if __name__ == "__main__":
    main()
