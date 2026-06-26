"""
app/tools/text_processor.py — MD/TXT → чанкинг → pgvector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from app.tools.doc_indexer import stream_index_text


async def stream_process_text_file(
    file_path: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    session_id: str | None = None,
    scope: str = "global",
) -> AsyncIterator[dict[str, Any]]:
    """Потоковая обработка текстового файла (.md, .txt)."""
    name = source_name or Path(file_path).name

    yield {"phase": "read", "message": f"Чтение «{name}»…"}

    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"Файл пуст: {name}")

    async for ev in stream_index_text(
        text, doc_type, name, db, pages=1, session_id=session_id, scope=scope
    ):
        yield ev


async def process_text_file(
    file_path: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    session_id: str | None = None,
    scope: str = "global",
) -> dict:
    """Полный пайплайн обработки MD/TXT (блокирующая обёртка)."""
    result: dict = {"pages": 1, "chunks": 0, "chunk_ids": []}
    async for ev in stream_process_text_file(
        file_path, doc_type, source_name, db, session_id=session_id, scope=scope
    ):
        if ev.get("phase") == "done":
            result = ev["result"]
    return result
