"""Git-коннектор: история коммитов + анализ импортов Java (tree-sitter).

Разбит на чистые функции (тестируются без git и без сети) и оркестратор
`GitConnector.extract`, который читает реальный репозиторий и пишет JSONL.

Производит узлы Commit/File/Module и рёбра:
  Commit -TOUCHES-> File
  File   -IMPORTS-> File        (импорт разрешился в файл того же репо)
  Module -DEPENDS_ON-> Module   (агрегация межмодульных импортов)
  Commit -MENTIONS-> Task       (ссылка KAFKA-1234 в сообщении)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.intermediate import JsonlWriter, edge, node

# Разделитель записей git log — маловероятная последовательность.
# REC_SEP ставится ПРЕФИКСОМ каждого коммита: с --name-only файлы идут после
# формата, поэтому маркер начала записи должен предшествовать заголовку, иначе
# список файлов прилипнет к заголовку следующего коммита.
_REC_SEP = "\x1e"
_FIELD_SEP = "\x1f"
_LOG_FORMAT = _REC_SEP + _FIELD_SEP.join(["%H", "%an", "%ae", "%aI", "%s"])

ISSUE_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
_IMPORT_RE_FALLBACK = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)


@dataclass
class Commit:
    sha: str
    author_name: str
    author_email: str
    date: str
    message: str
    files: list[str] = field(default_factory=list)

    @property
    def issue_refs(self) -> list[str]:
        return sorted(set(ISSUE_RE.findall(self.message)))


def parse_git_log(raw: str) -> list[Commit]:
    """Разбирает вывод `git log` в машинном формате (см. _LOG_FORMAT + --name-only).

    Чистая функция — не требует git. Формат каждой записи:
      <sha>\x1f<an>\x1f<ae>\x1f<date>\x1f<subject>\x1e\n<file>\n<file>...
    """
    commits: list[Commit] = []
    for chunk in raw.split(_REC_SEP):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = chunk.split("\n")
        header = lines[0]
        parts = header.split(_FIELD_SEP)
        if len(parts) < 5:
            continue
        sha, an, ae, date, subject = parts[:5]
        files = [ln.strip() for ln in lines[1:] if ln.strip()]
        commits.append(
            Commit(sha=sha, author_name=an, author_email=ae, date=date, message=subject, files=files)
        )
    return commits


def extract_imports(java_source: str) -> list[str]:
    """Извлекает fully-qualified имена из `import ...;` Java-файла.

    Пытается через tree-sitter (точнее), с регексным фолбэком.
    """
    try:
        return _extract_imports_treesitter(java_source)
    except Exception:
        return sorted(set(_IMPORT_RE_FALLBACK.findall(java_source)))


def _extract_imports_treesitter(java_source: str) -> list[str]:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser

    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(java_source.encode("utf-8"))
    imports: set[str] = set()

    def walk(node_):
        if node_.type == "import_declaration":
            text = java_source[node_.start_byte : node_.end_byte]
            m = _IMPORT_RE_FALLBACK.match(text if text.startswith("import") else "import " + text)
            # надёжнее — вытащить именованный узел scoped_identifier/identifier
            for child in node_.children:
                if child.type in ("scoped_identifier", "identifier"):
                    imports.add(java_source[child.start_byte : child.end_byte])
                    break
            if m:
                imports.add(m.group(1))
        for child in node_.children:
            walk(child)

    walk(tree.root_node)
    return sorted(imports)


def module_of(path: str, components: list[str] | None = None) -> str | None:
    """Топ-уровневый компонент из пути ('clients/src/.../Foo.java' -> 'clients').

    Если задан список components, возвращает модуль только для них.
    """
    parts = Path(path).as_posix().split("/")
    if not parts:
        return None
    top = parts[0]
    if components and top not in components:
        return None
    return top


def fqn_to_path_suffix(fqn: str) -> str:
    """org.apache.kafka.clients.Foo -> clients/Foo.java (грубое сопоставление)."""
    return fqn.replace(".", "/") + ".java"


@dataclass
class GitConnector:
    repo_path: str
    components: list[str] = field(default_factory=list)
    since: str | None = None

    def _run_git_log(self) -> str:
        cmd = ["git", "-C", self.repo_path, "log", f"--pretty=format:{_LOG_FORMAT}", "--name-only"]
        if self.since:
            cmd.insert(4, f"--since={self.since}")
        # git выводит UTF-8; text=True декодировал бы локалью (cp1251 на Windows)
        # и падал на не-ASCII в сообщениях/именах. Форсируем UTF-8.
        return subprocess.run(
            cmd, capture_output=True, check=True, encoding="utf-8", errors="replace"
        ).stdout

    def _java_files(self) -> list[str]:
        root = Path(self.repo_path)
        result: list[str] = []
        for p in root.rglob("*.java"):
            rel = p.relative_to(root).as_posix()
            if not self.components or module_of(rel, self.components):
                result.append(rel)
        return result

    def extract(self, out_path: str) -> dict:
        """Читает репозиторий и пишет JSONL. Возвращает статистику."""
        commits = parse_git_log(self._run_git_log())
        java_files = self._java_files()
        return self._emit(commits, java_files, out_path)

    def _emit(self, commits: list[Commit], java_files: list[str], out_path: str) -> dict:
        """Строит записи из уже разобранных данных (тестируемо без git)."""
        # Индекс: суффикс пути -> реальный путь, для разрешения импортов.
        by_suffix: dict[str, str] = {}
        for f in java_files:
            by_suffix[Path(f).name] = f

        module_deps: set[tuple[str, str]] = set()
        stats = {"commits": 0, "files": 0, "imports": 0, "module_deps": 0, "mentions": 0}

        with JsonlWriter(out_path) as w:
            # Файлы и модули
            seen_modules: set[str] = set()
            for f in java_files:
                src = {"source": "git", "uri": f}
                w.write(node("File", f"file:{f}", {"path": f}, src))
                stats["files"] += 1
                mod = module_of(f, self.components)
                if mod and mod not in seen_modules:
                    seen_modules.add(mod)
                    w.write(node("Module", f"module:{mod}", {"name": mod}, {"source": "git"}))

            # Импорты между файлами + агрегация модулей
            root = Path(self.repo_path)
            for f in java_files:
                fp = root / f
                try:
                    source = fp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                src_mod = module_of(f, self.components)
                for imp in extract_imports(source):
                    target = self._resolve_import(imp, by_suffix)
                    if not target:
                        continue
                    w.write(edge("IMPORTS", f"file:{f}", f"file:{target}", source={"source": "git"}))
                    stats["imports"] += 1
                    tgt_mod = module_of(target, self.components)
                    if src_mod and tgt_mod and src_mod != tgt_mod:
                        module_deps.add((src_mod, tgt_mod))

            for a, b in sorted(module_deps):
                w.write(edge("DEPENDS_ON", f"module:{a}", f"module:{b}", source={"source": "git"}))
                stats["module_deps"] += 1

            # Коммиты, касания файлов, упоминания тикетов
            for c in commits:
                src = {"source": "git", "date": c.date, "author": c.author_name}
                w.write(
                    node(
                        "Commit",
                        f"commit:{c.sha}",
                        {"sha": c.sha, "author": c.author_name, "date": c.date, "message": c.message},
                        src,
                    )
                )
                stats["commits"] += 1
                for f in c.files:
                    if f.endswith(".java") and (not self.components or module_of(f, self.components)):
                        w.write(edge("TOUCHES", f"commit:{c.sha}", f"file:{f}", source={"source": "git"}))
                for ref in c.issue_refs:
                    w.write(edge("MENTIONS", f"commit:{c.sha}", f"task:{ref}", source={"source": "git"}))
                    stats["mentions"] += 1

        return stats

    @staticmethod
    def _resolve_import(fqn: str, by_suffix: dict[str, str]) -> str | None:
        """Разрешает импорт в файл репозитория по имени класса (последний сегмент)."""
        cls = fqn.split(".")[-1]
        return by_suffix.get(f"{cls}.java")
