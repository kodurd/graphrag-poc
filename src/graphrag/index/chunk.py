"""Чанкинг текста — чистые функции (без Neo4j и эмбеддингов)."""

from __future__ import annotations


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


def plan_chunks(
    nodes: list[tuple[str, str, str | None]], size: int = 800, overlap: int = 120
) -> list[dict]:
    """Строит спецификации чанков из (parent_id, text, uri). Чистая функция.

    id чанка: 'chunk:<parent_id>#<seq>'.
    """
    specs: list[dict] = []
    for parent_id, text, uri in nodes:
        for seq, piece in enumerate(chunk_text(text, size, overlap)):
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
