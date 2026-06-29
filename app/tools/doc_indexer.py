"""
app/tools/doc_indexer.py — чанкинг и индексация текста в pgvector (parent-child).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Literal

from app.agents.llm_client import embed_texts
from app.config import get_settings
from app.tools.markdown_chunker import (
    StructuralSection,
    has_markdown_headings,
    legacy_recursive_sections,
    markdown_structural_chunk,
)

settings = get_settings()

DocScope = Literal["global", "session"]

_MARKDOWN_DOC_TYPES = frozenset({"iec_standard", "elbrus_manual"})


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


def _use_markdown_chunking(source_name: str, doc_type: str, full_text: str) -> bool:
    mode = settings.chunking_mode.lower()
    if mode == "recursive":
        return False
    if mode == "markdown":
        return True
    if Path(source_name).suffix.lower() == ".md":
        return True
    if doc_type in _MARKDOWN_DOC_TYPES and has_markdown_headings(full_text):
        return True
    return False


def _chunk_document(full_text: str, source_name: str, doc_type: str) -> list[StructuralSection]:
    ps = settings.chunk_parent_size
    po = settings.chunk_parent_overlap
    cs = settings.chunk_child_size
    co = settings.chunk_child_overlap
    pl = settings.chunk_markdown_parent_level

    if _use_markdown_chunking(source_name, doc_type, full_text):
        sections = markdown_structural_chunk(
            full_text,
            parent_level=pl,
            parent_max_chars=ps,
            child_size=cs,
            child_overlap=co,
        )
        if sections:
            return sections
    return legacy_recursive_sections(full_text, ps, po, cs, co)


def _parent_placeholder_embedding(child_embs: list[list[float]]) -> list[float]:
    """Parent не участвует в dense search — копируем первый child embedding для NOT NULL."""
    if child_embs:
        return list(child_embs[0])
    dims = settings.embedding_dimensions
    return [0.0] * dims


async def stream_index_text(
    full_text: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    pages: int = 1,
    session_id: str | None = None,
    scope: DocScope = "global",
) -> AsyncIterator[dict[str, Any]]:
    """Индексация текста: chunk → embed → pgvector."""
    if db is None:
        chunks = recursive_chunk(full_text, settings.chunk_size, settings.chunk_overlap)
        result = {"pages": pages, "chunks": len(chunks), "chunk_ids": []}
        yield {"phase": "done", "result": result}
        return

    from app.memory.store import add_parent_child, delete_document

    await delete_document(db, source_name, scope, session_id, commit=False)

    sections = _chunk_document(full_text, source_name, doc_type)
    total_parents = len(sections)
    yield {
        "phase": "chunk",
        "message": f"Разбиение на {total_parents} блоков для векторизации…",
        "total": total_parents,
    }

    chunk_ids: list[int] = []
    processed = 0

    try:
        for pi, section in enumerate(sections):
            child_texts = [c.content for c in section.children]
            if not child_texts:
                continue

            child_embs = await embed_texts(child_texts)
            parent_emb = _parent_placeholder_embedding(child_embs)
            children = [
                {
                    "content": c.content,
                    "embedding": e,
                    "index": ci,
                    "metadata": {
                        "heading_path": c.heading_path,
                        "content_type": c.content_type,
                    },
                }
                for ci, (c, e) in enumerate(zip(section.children, child_embs))
            ]

            meta: dict[str, Any] = {
                "type": "doc",
                "scope": scope,
                "doc_type": doc_type,
                "source": source_name,
                "parent_index": pi,
                "chunking": section.chunking,
                "heading_path": section.heading_path,
                "heading_level": section.heading_level,
                "content_types": section.content_types,
            }
            if scope == "session" and session_id:
                meta["session_id"] = session_id

            _, cids = await add_parent_child(
                db, section.parent_text, parent_emb, children, meta, commit=False
            )
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

        await db.commit()
    except Exception:
        await db.rollback()
        raise

    result = {"pages": pages, "chunks": len(chunk_ids), "chunk_ids": chunk_ids}
    yield {"phase": "done", "result": result}


async def index_text(
    full_text: str,
    doc_type: str = "general",
    source_name: str = "",
    db=None,
    pages: int = 1,
    session_id: str | None = None,
    scope: DocScope = "global",
) -> dict:
    """Блокирующая обёртка над stream_index_text."""
    result: dict = {"pages": pages, "chunks": 0, "chunk_ids": []}
    async for ev in stream_index_text(
        full_text,
        doc_type,
        source_name or "document",
        db,
        pages=pages,
        session_id=session_id,
        scope=scope,
    ):
        if ev.get("phase") == "done":
            result = ev["result"]
    return result
