"""
app/tools/doc_indexer.py — чанкинг и индексация текста в pgvector (parent-child).
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from app.agents.llm_client import embed_texts
from app.config import get_settings

settings = get_settings()


def recursive_chunk(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Простой рекурсивный чанкер."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        parts = text.split(sep) if sep else list(text)
        if len(parts) > 1:
            chunks, current = [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = current[-overlap:] + sep + part if overlap and current else part
            if current:
                chunks.append(current)
            result = [c.strip() for c in chunks if c.strip()]
            if result:
                return result

    return [text[:size]]


async def stream_index_text(
    full_text: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    pages: int = 1,
) -> AsyncIterator[dict[str, Any]]:
    """Индексация текста: chunk → embed → pgvector."""
    if db is None:
        chunks = recursive_chunk(full_text, settings.chunk_size, settings.chunk_overlap)
        result = {"pages": pages, "chunks": len(chunks), "chunk_ids": []}
        yield {"phase": "done", "result": result}
        return

    parent_chunks = recursive_chunk(full_text, 1500, 150)
    total_parents = len([p for p in parent_chunks if p.strip()])
    yield {
        "phase": "chunk",
        "message": f"Разбиение на {total_parents} блоков для векторизации…",
        "total": total_parents,
    }

    chunk_ids: list[int] = []
    processed = 0

    for pi, parent_text in enumerate(parent_chunks):
        if not parent_text.strip():
            continue

        child_texts = recursive_chunk(parent_text, 800, 80)
        if not child_texts:
            continue

        parent_emb_list = await embed_texts([parent_text])
        parent_emb = parent_emb_list[0]
        child_embs = await embed_texts(child_texts)
        children = [
            {"content": t, "embedding": e, "index": ci}
            for ci, (t, e) in enumerate(zip(child_texts, child_embs))
        ]

        meta = {
            "type": "doc",
            "doc_type": doc_type,
            "source": source_name,
            "parent_index": pi,
        }

        from app.memory.store import add_parent_child

        _, cids = await add_parent_child(db, parent_text, parent_emb, children, meta)
        chunk_ids.extend(cids)

        processed += 1
        pct = int(processed / total_parents * 100) if total_parents else 100
        yield {
            "phase": "embed",
            "current": processed,
            "total": total_parents,
            "pct": pct,
            "message": f"Векторизация: {pct}% ({processed}/{total_parents} блоков)",
        }

    result = {"pages": pages, "chunks": len(chunk_ids), "chunk_ids": chunk_ids}
    yield {"phase": "done", "result": result}


async def index_text(
    full_text: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    pages: int = 1,
) -> dict:
    """Блокирующая обёртка над stream_index_text."""
    result: dict = {"pages": pages, "chunks": 0, "chunk_ids": []}
    async for ev in stream_index_text(
        full_text, doc_type, source_name or "document", db, pages=pages
    ):
        if ev.get("phase") == "done":
            result = ev["result"]
    return result
