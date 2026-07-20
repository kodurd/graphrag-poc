"""Главный сценарий: «лог с ошибкой → что затронуто».

Пайплайн:
  1. извлечь сущности из лога (регекс + опционально LLM)
  2. сматчить в узлы графа (модули/файлы)
  3. Cypher-обход зависимостей 1..N хопов -> затронутые модули + владельцы
  4. подтянуть связанные тикеты («уже чинили») и страницы вики
"""

from __future__ import annotations

import re

from graphrag.graph.connection import Neo4jConnection
from graphrag.llm.base import LLMClient

_EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Exception|Error))\b")
_FRAME_RE = re.compile(r"\bat\s+([\w.$]+)\(")
_KAFKA_FQN_RE = re.compile(r"\borg\.apache\.kafka\.([a-z][a-z0-9]*)")

# Пакеты org.apache.kafka.*, которые не являются осмысленными «модулями» для
# impact-анализа: слишком широкие (встречаются почти в каждом трейсе) и
# матчились бы на всё. Отсеиваем до вывода.
# NB: блок-лист по имени — если такой модуль реально появится в графе, он станет
# невидим для impact. Для PoC-корпуса (clients/connect/streams) это безопасно;
# при росте корпуса заменить на позитивный allowlist известных модулей.
_MODULE_STOPLIST = frozenset({"common"})

_EXTRACT_PROMPT = (
    "Из фрагмента лога извлеки сущности. Верни JSON с ключами "
    '"services", "modules", "exceptions" (списки строк). Только JSON.\n\nЛОГ:\n'
)


def extract_entities(log_text: str, llm: LLMClient | None = None) -> dict:
    """Извлекает {exceptions, classes, modules} из лога.

    Регекс — основной путь (детерминированный). LLM (если дан) добавляет редкие
    сущности, которые регексом не взять; результаты объединяются.
    """
    exceptions = set(_EXC_RE.findall(log_text))
    classes = set(_FRAME_RE.findall(log_text))
    # Модули берём только из FQN org.apache.kafka.<module>. Содержимое [...] —
    # это имя потока (worker-1, main, task-connect-sink-0), а не модуль.
    modules = set(_KAFKA_FQN_RE.findall(log_text))

    if llm is not None:
        try:
            extra = llm.extract_json(_EXTRACT_PROMPT + log_text[:4000])
            if isinstance(extra, dict):
                modules |= {str(m).lower() for m in extra.get("modules", []) or []}
                modules |= {str(m).lower() for m in extra.get("services", []) or []}
                exceptions |= {str(e) for e in extra.get("exceptions", []) or []}
        except Exception:
            pass  # LLM — необязательный усилитель; регекс уже дал результат

    modules -= _MODULE_STOPLIST

    return {
        "exceptions": sorted(exceptions),
        "classes": sorted(classes),
        "modules": sorted(modules),
    }


def _module_of_file_id(file_id: str) -> str | None:
    # file:clients/src/.../Foo.java -> clients
    path = file_id.split(":", 1)[-1]
    parts = path.split("/")
    return parts[0] if parts and parts[0] else None


class ImpactAnalyzer:
    """Связывает лог с графом и собирает подграф затронутого."""

    def __init__(
        self,
        conn: Neo4jConnection,
        *,
        llm: LLMClient | None = None,
        max_hops: int = 3,
    ):
        self.conn = conn
        self.llm = llm
        self.max_hops = max(1, min(int(max_hops), 5))  # безопасный предел обхода

    def match_modules(self, entities: dict) -> list[str]:
        """Сопоставляет сущности лога с существующими узлами Module (по имени)."""
        candidates = {m.lower() for m in entities.get("modules", [])}

        # Модули затронутых файлов (по имени класса из стектрейса).
        for cls in entities.get("classes", []):
            simple = cls.split(".")[-1].split("$")[0]
            rows = self.conn.run(
                "MATCH (f:File) WHERE f.id ENDS WITH $suffix RETURN f.id AS id",
                suffix=f"/{simple}.java",
            )
            for r in rows:
                mod = _module_of_file_id(r["id"])
                if mod:
                    candidates.add(mod.lower())

        if not candidates:
            return []
        existing = self.conn.run(
            "MATCH (m:Module) WHERE toLower(m.name) IN $names RETURN m.id AS id",
            names=sorted(candidates),
        )
        return [r["id"] for r in existing]

    def impact_subgraph(self, failing_ids: list[str]) -> dict:
        """Обходит DEPENDS_ON от упавших модулей и собирает связанное."""
        if not failing_ids:
            return {
                "failing": [],
                "affected_modules": [],
                "owners": [],
                "related_tasks": [],
                "related_pages": [],
            }

        hops = self.max_hops
        # Затронутые = те, кто (транзитивно) зависит от упавшего модуля.
        affected = self.conn.run(
            f"""
            MATCH (failing:Module) WHERE failing.id IN $ids
            MATCH (affected:Module)-[:DEPENDS_ON*1..{hops}]->(failing)
            RETURN DISTINCT affected.id AS id, affected.name AS name
            ORDER BY name
            """,
            ids=failing_ids,
        )
        affected_ids = [r["id"] for r in affected]
        all_module_ids = list(dict.fromkeys(failing_ids + affected_ids))

        # Владельцы: люди, назначенные на тикеты, упоминающие эти модули.
        owners = self.conn.run(
            """
            MATCH (t:Task)-[:MENTIONS]->(m:Module) WHERE m.id IN $ids
            MATCH (t)-[:ASSIGNED_TO]->(p:Person)
            RETURN DISTINCT p.name AS name, m.name AS module
            ORDER BY name
            """,
            ids=all_module_ids,
        )

        # «Уже чинили»: тикеты, упоминающие затронутые модули (+ их дубли/фиксы).
        related_tasks = self.conn.run(
            """
            MATCH (t:Task)-[:MENTIONS]->(m:Module) WHERE m.id IN $ids
            OPTIONAL MATCH (t)-[:DUPLICATES|FIXED_BY]->(rel:Task)
            WITH collect(DISTINCT t) + collect(DISTINCT rel) AS tasks
            UNWIND tasks AS task
            // Отсекаем узлы-заглушки: git создаёт Task из ссылок KAFKA-xxxx в
            // коммитах, у которых нет данных JIRA (ни key, ни summary, ни uri).
            WITH task WHERE task IS NOT NULL AND task.key IS NOT NULL
            RETURN DISTINCT task.key AS key, task.summary AS summary,
                   task.status AS status, task.uri AS uri
            ORDER BY key
            """,
            ids=all_module_ids,
        )

        # Страницы вики, упоминающие связанные тикеты.
        related_pages = self.conn.run(
            """
            MATCH (t:Task)-[:MENTIONS]->(m:Module) WHERE m.id IN $ids
            MATCH (page:Page)-[:MENTIONS]->(t)
            RETURN DISTINCT page.title AS title, page.uri AS uri
            ORDER BY title
            """,
            ids=all_module_ids,
        )

        return {
            "failing": failing_ids,
            "affected_modules": affected,
            "owners": owners,
            "related_tasks": related_tasks,
            "related_pages": related_pages,
        }

    def analyze(self, log_text: str) -> dict:
        """Полный проход: лог -> сущности -> матчинг -> подграф."""
        entities = extract_entities(log_text, self.llm)
        failing = self.match_modules(entities)
        result = self.impact_subgraph(failing)
        result["entities"] = entities
        return result
