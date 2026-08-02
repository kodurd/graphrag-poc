"""Git-коннектор — чистые парсеры и сборка записей на фикстурах."""

from __future__ import annotations

from pathlib import Path

from graphrag.connectors.git import (
    Commit,
    GitConnector,
    extract_imports,
    module_of,
    parse_git_bodies,
    parse_git_log,
)
from graphrag.connectors.git import _FIELD_SEP, _REC_SEP
from graphrag.intermediate import JsonlWriter, read_jsonl


def _mk_bodies(*records: tuple[str, str]) -> str:
    """Сырой вывод git log формата _BODY_FORMAT: REC_SEP + sha + FIELD_SEP + %B."""
    return "".join(_REC_SEP + sha + _FIELD_SEP + body for sha, body in records)


def _mk_log(*records: tuple[str, str, str, str, str, list[str]]) -> str:
    """Собирает сырой git-log в машинном формате из кортежей.

    Повторяет реальный вывод git: REC_SEP-префикс, заголовок, затем файлы.
    """
    out = []
    for sha, an, ae, date, subj, files in records:
        header = _REC_SEP + _FIELD_SEP.join([sha, an, ae, date, subj])
        out.append(header + "\n" + "\n".join(files))
    return "\n".join(out)


# --- parse_git_log ---

def test_parse_git_log_happy():
    raw = _mk_log(
        ("abc123", "Ann Dev", "ann@x.io", "2024-03-01T10:00:00+00:00", "KAFKA-101 fix producer",
         ["clients/src/Producer.java", "README.md"]),
        ("def456", "Bob Dev", "bob@x.io", "2024-03-02T11:00:00+00:00", "refactor utils", []),
    )
    commits = parse_git_log(raw)
    assert len(commits) == 2
    assert commits[0].sha == "abc123"
    assert commits[0].author_name == "Ann Dev"
    assert "clients/src/Producer.java" in commits[0].files


def test_commit_issue_refs():
    c = Commit("s", "a", "a@x", "d", "KAFKA-101 and KAFKA-205 fixed")
    assert c.issue_refs == ["KAFKA-101", "KAFKA-205"]


def test_commit_without_issue_ref_is_empty():
    """Коммит без KAFKA-xxx → пустой список упоминаний, не падение."""
    c = Commit("s", "a", "a@x", "d", "just a refactor")
    assert c.issue_refs == []


# --- parse_git_bodies + тело коммита (U1) ---

def test_parse_git_bodies_multiline():
    raw = _mk_bodies(
        ("abc123", "KAFKA-1 fix producer\n\nDetailed body line one.\nLine two."),
        ("def456", "refactor utils"),  # subject-only, тело пустое
    )
    bodies = parse_git_bodies(raw)
    assert bodies["abc123"] == "KAFKA-1 fix producer\n\nDetailed body line one.\nLine two."
    assert bodies["def456"] == "refactor utils"
    # тело начинается с subject (%B) — тугой сигнал в начале
    assert bodies["abc123"].splitlines()[0] == "KAFKA-1 fix producer"


def test_commit_subject_vs_body_refs():
    """subject_issue_refs = только заголовок; issue_refs = весь message (subject+body)."""
    c = Commit("s", "a", "a@x", "d",
               message="KAFKA-1 fix\n\nReverts KAFKA-2, follow-up to KAFKA-3",
               subject="KAFKA-1 fix")
    assert c.subject_issue_refs == ["KAFKA-1"]
    assert c.issue_refs == ["KAFKA-1", "KAFKA-2", "KAFKA-3"]


def test_commit_subject_defaults_to_message():
    """Без явного subject (как из parse_git_log) subject = message."""
    c = Commit("s", "a", "a@x", "d", "KAFKA-9 fix")
    assert c.subject == "KAFKA-9 fix"
    assert c.subject_issue_refs == ["KAFKA-9"]


def test_emit_commit_tags_body_mentions(tmp_path: Path):
    """MENTIONS из тела помечены ref_source='body'; из заголовка — 'subject'.
    commits-only (with_touches=False) не пишет File/Module/TOUCHES."""
    c = Commit("abc", "Ann", "ann@x", "2024-03-01T10:00:00+00:00",
               message="KAFKA-1 fix\n\nReverts KAFKA-2",
               subject="KAFKA-1 fix",
               files=["clients/src/Producer.java"])
    conn = GitConnector(str(tmp_path))
    out = tmp_path / "commits.jsonl"
    stats = {"commits": 0, "mentions": 0, "body_mentions": 0}
    with JsonlWriter(str(out)) as w:
        conn._emit_commit(w, c, stats, with_touches=False)

    records = list(read_jsonl(out))
    edges = {r["to"]: r for r in records if r["kind"] == "edge" and r["type"] == "MENTIONS"}
    assert edges["task:KAFKA-1"]["props"]["ref_source"] == "subject"
    assert edges["task:KAFKA-2"]["props"]["ref_source"] == "body"
    assert stats["body_mentions"] == 1
    # commits-only: только Commit-узел, без File/Module/TOUCHES
    assert not any(r["kind"] == "node" and r["label"] in ("File", "Module") for r in records)
    assert not any(r["kind"] == "edge" and r["type"] == "TOUCHES" for r in records)
    # message коммита несёт тело
    commit_node = next(r for r in records if r["kind"] == "node" and r["label"] == "Commit")
    assert "Reverts KAFKA-2" in commit_node["props"]["message"]


# --- extract_imports (tree-sitter с фолбэком) ---

def test_extract_imports_from_java():
    src = """
    package org.apache.kafka.clients;
    import org.apache.kafka.common.Utils;
    import static java.util.Objects.requireNonNull;
    public class Producer {}
    """
    imps = extract_imports(src)
    assert "org.apache.kafka.common.Utils" in imps


def test_module_of_respects_components():
    assert module_of("clients/src/Foo.java", ["clients", "common"]) == "clients"
    assert module_of("core/src/Bar.java", ["clients", "common"]) is None


# --- сборка записей (без git) ---

def test_emit_builds_imports_and_cross_module_depends_on(tmp_path: Path):
    # Фикстура: два модуля, межмодульный импорт clients -> common
    (tmp_path / "clients/src").mkdir(parents=True)
    (tmp_path / "common/src").mkdir(parents=True)
    (tmp_path / "clients/src/Producer.java").write_text(
        "package org.apache.kafka.clients;\n"
        "import org.apache.kafka.common.Utils;\n"
        "public class Producer {}\n",
        encoding="utf-8",
    )
    (tmp_path / "common/src/Utils.java").write_text(
        "package org.apache.kafka.common;\npublic class Utils {}\n", encoding="utf-8"
    )

    java_files = ["clients/src/Producer.java", "common/src/Utils.java"]
    commits = [
        Commit("abc", "Ann", "ann@x", "2024-03-01T10:00:00+00:00",
               "KAFKA-101 fix", ["clients/src/Producer.java", "docs/readme.md"])
    ]
    conn = GitConnector(str(tmp_path), components=["clients", "common"])
    out = tmp_path / "git.jsonl"
    stats = conn._emit(commits, java_files, str(out))

    records = list(read_jsonl(out))
    edges = [r for r in records if r["kind"] == "edge"]

    imports = [e for e in edges if e["type"] == "IMPORTS"]
    assert any(e["from"] == "file:clients/src/Producer.java"
               and e["to"] == "file:common/src/Utils.java" for e in imports)

    depends = [e for e in edges if e["type"] == "DEPENDS_ON"]
    assert ("module:clients", "module:common") in {(e["from"], e["to"]) for e in depends}

    mentions = [e for e in edges if e["type"] == "MENTIONS"]
    assert any(e["to"] == "task:KAFKA-101" for e in mentions)

    # TOUCHES только для java-файлов; docs/readme.md пропущен без падения
    touches = [e for e in edges if e["type"] == "TOUCHES"]
    touched = {e["to"] for e in touches}
    assert "file:clients/src/Producer.java" in touched
    assert "file:docs/readme.md" not in touched

    assert stats["commits"] == 1 and stats["imports"] >= 1 and stats["module_deps"] == 1


def test_emit_dangling_import_is_skipped(tmp_path: Path):
    """Импорт внешнего класса (нет файла в репо) не создаёт IMPORTS-ребро."""
    (tmp_path / "clients/src").mkdir(parents=True)
    (tmp_path / "clients/src/Producer.java").write_text(
        "import java.util.List;\npublic class Producer {}\n", encoding="utf-8"
    )
    java_files = ["clients/src/Producer.java"]
    conn = GitConnector(str(tmp_path), components=["clients"])
    out = tmp_path / "git.jsonl"
    conn._emit([], java_files, str(out))
    edges = [r for r in read_jsonl(out) if r["kind"] == "edge" and r["type"] == "IMPORTS"]
    assert edges == []
