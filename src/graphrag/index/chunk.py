"""Чанкинг текста — чистые функции (без Neo4j и эмбеддингов)."""

from __future__ import annotations

import re

# Сентинел раздела KIP-страницы: строка, начинающаяся с "## " (маркер из
# parse_page, U1). Детектор — есть ли хоть один раздел; сплиттер режет текст
# ПЕРЕД каждым таким маркером, сохраняя заголовок в теле секции.
_SECTION_RE = re.compile(r"(?m)^## ")
_SECTION_SPLIT_RE = re.compile(r"(?m)(?=^## )")


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Режет текст на окна `size` символов с перекрытием `overlap`.

    Пустой/пробельный текст → []. Короткий (<= size) → один чанк.
    """
    if overlap >= size:
        raise ValueError("overlap должен быть меньше size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunk = text[start : start + size]
        chunks.append(chunk)
        if start + size >= len(text):
            break
        start += step
    return chunks


def _plan_pieces(parent_id: str, text: str, size: int, overlap: int) -> list[str]:
    """Куски текста узла с учётом источника.

    Дискриминатор источника — префикс `parent_id` (`page:` у KIP-страниц,
    `task:`/`commit:` у тикетов и коммитов), он уже несёт нужный сигнал, поэтому
    отдельный флаг в кортеж узла не пробрасываем (см. vector.py).

    Page-узлы с секционными маркерами `## ` (из U1) режем ПО РАЗДЕЛАМ: короткий
    раздел = один цельный чанк, длинный (> size) — окном ВНУТРИ раздела, никогда
    не смешивая соседние. Тикеты/коммиты и Page без маркеров (загруженные до U1)
    идут прежним слепым оконным путём — байт-идентично старому поведению.
    """
    if parent_id.startswith("page:") and _SECTION_RE.search(text):
        pieces: list[str] = []
        for section in _SECTION_SPLIT_RE.split(text):
            if section.strip():
                pieces.extend(chunk_text(section, size, overlap))
        return pieces
    return chunk_text(text, size, overlap)


def plan_chunks(
    nodes: list[tuple[str, str, str | None]], size: int = 800, overlap: int = 120
) -> list[dict]:
    """Строит спецификации чанков из (parent_id, text, uri). Чистая функция.

    id чанка: 'chunk:<parent_id>#<seq>'. Для Page-узлов режет по разделам
    (см. `_plan_pieces`); seq сквозной по всем разделам страницы.
    """
    specs: list[dict] = []
    for parent_id, text, uri in nodes:
        for seq, piece in enumerate(_plan_pieces(parent_id, text, size, overlap)):
            specs.append(
                {
                    "id": f"chunk:{parent_id}#{seq}",
                    "parent": parent_id,
                    "text": piece,
                    "uri": uri or "",
                    "seq": seq,
                }
            )
    return specs
