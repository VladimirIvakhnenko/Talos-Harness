"""
app/memory/store.py — Векторная память: dense search + parent-child docs + chat messages.
"""
from __future__ import annotations
import json
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

settings = get_settings()

MemType = Literal["doc", "chat"]
DocScope = Literal["global", "session"]
ChunkLvl = Literal["child", "parent", "single"]


def _embedding_expr(column: str = "embedding") -> str:
    dims = settings.embedding_dimensions
    if dims > 2000:
        return f"({column}::halfvec({dims}))"
    return column


def _query_embedding_cast() -> str:
    dims = settings.embedding_dimensions
    if dims > 2000:
        return f"CAST(:emb AS halfvec({dims}))"
    return "CAST(:emb AS vector)"


async def add_memory(
    db: AsyncSession,
    content: str,
    embedding: list[float],
    metadata: dict,
    chunk_level: ChunkLvl = "single",
    parent_id: int | None = None,
) -> int:
    r = await db.execute(
        text("""
        INSERT INTO memories (content, embedding, metadata, chunk_level, parent_id)
        VALUES (:c, CAST(:e AS vector), CAST(:m AS jsonb), :l, :p) RETURNING id
    """),
        {
            "c": content,
            "e": str(embedding),
            "m": json.dumps(metadata),
            "l": chunk_level,
            "p": parent_id,
        },
    )
    await db.commit()
    return r.scalar_one()


async def add_parent_child(
    db: AsyncSession,
    parent_content: str,
    parent_emb: list[float],
    children: list[dict],
    metadata: dict,
) -> tuple[int, list[int]]:
    pid = await add_memory(
        db, parent_content, parent_emb, {**metadata, "chunk_level": "parent"}, "parent"
    )
    cids = []
    for c in children:
        cid = await add_memory(
            db,
            c["content"],
            c["embedding"],
            {**metadata, "chunk_level": "child", "chunk_index": c.get("index", 0)},
            "child",
            pid,
        )
        cids.append(cid)
    return pid, cids


async def delete_document(
    db: AsyncSession,
    source: str,
    scope: DocScope,
    session_id: str | None = None,
) -> int:
    """Удалить документ по source (parent + children через CASCADE)."""
    if scope == "global":
        r = await db.execute(
            text("""
                DELETE FROM memories
                WHERE metadata->>'type' = 'doc'
                  AND metadata->>'scope' = 'global'
                  AND metadata->>'source' = :src
                  AND chunk_level = 'parent'
            """),
            {"src": source},
        )
    else:
        r = await db.execute(
            text("""
                DELETE FROM memories
                WHERE metadata->>'type' = 'doc'
                  AND metadata->>'scope' = 'session'
                  AND metadata->>'session_id' = :sid
                  AND metadata->>'source' = :src
                  AND chunk_level = 'parent'
            """),
            {"src": source, "sid": session_id},
        )
    await db.commit()
    return r.rowcount


async def dense_search(
    db: AsyncSession,
    emb: list[float],
    top_k: int = 20,
    filter_type: MemType | None = None,
    session_id: str | None = None,
    chunk_level: ChunkLvl | None = "child",
    doc_scope_filter: str | None = None,
) -> list[dict]:
    conds, params = ["1=1"], {"emb": str(emb), "top_k": top_k}
    if filter_type:
        conds.append("metadata->>'type' = :ft")
        params["ft"] = filter_type
    if chunk_level:
        conds.append("chunk_level = :lvl")
        params["lvl"] = chunk_level
    if doc_scope_filter == "documents":
        conds.append(
            "(metadata->>'scope' = 'global' OR "
            "(metadata->>'scope' = 'session' AND metadata->>'session_id' = :sid))"
        )
        params["sid"] = session_id or ""
    elif session_id:
        conds.append("metadata->>'session_id' = :sid")
        params["sid"] = session_id
    emb_expr = _embedding_expr()
    emb_cast = _query_embedding_cast()
    rows = await db.execute(
        text(f"""
        SELECT id, content, metadata, chunk_level, parent_id,
               1-({emb_expr} <=> {emb_cast}) AS score
        FROM memories WHERE {" AND ".join(conds)}
        ORDER BY {emb_expr} <=> {emb_cast} LIMIT :top_k
    """),
        params,
    )
    return [dict(r._mapping) for r in rows.fetchall()]


async def lift_to_parents(db: AsyncSession, results: list[dict]) -> list[dict]:
    pids = list({d["parent_id"] for d in results if d.get("parent_id")})
    singles = [d for d in results if not d.get("parent_id")]
    if not pids:
        return singles
    rows = await db.execute(
        text("SELECT id, content, metadata, chunk_level, parent_id FROM memories WHERE id=ANY(:ids)"),
        {"ids": pids},
    )
    return [dict(r._mapping) for r in rows.fetchall()] + singles


async def search_documents(
    db: AsyncSession,
    query: str,
    emb: list[float],
    session_id: str,
    top_k: int = 5,
) -> list[dict]:
    """Dense-only RAG: global docs + docs текущей сессии, child → parent."""
    _ = query  # embedding carries semantic signal; query kept for API compat
    hits = await dense_search(
        db, emb, top_k * 4, filter_type="doc", chunk_level="child",
        doc_scope_filter="documents", session_id=session_id,
    )
    parents = await lift_to_parents(db, hits)
    seen: set[int] = set()
    unique: list[dict] = []
    for p in parents:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    return unique[:top_k]


async def save_chat_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    embedding: list[float],
    turn_index: int,
) -> int:
    return await add_memory(
        db,
        content,
        embedding,
        {
            "type": "chat",
            "session_id": session_id,
            "role": role,
            "turn_index": turn_index,
        },
        "single",
    )


async def load_chat_history(db: AsyncSession, session_id: str) -> list[dict]:
    """История для Gradio type=messages."""
    rows = await db.execute(
        text("""
            SELECT metadata->>'role' AS role, content
            FROM memories
            WHERE metadata->>'type' = 'chat'
              AND metadata->>'session_id' = :sid
            ORDER BY (metadata->>'turn_index')::int, created_at
        """),
        {"sid": session_id},
    )
    return [{"role": r.role, "content": r.content} for r in rows.fetchall()]


async def next_turn_index(db: AsyncSession, session_id: str) -> int:
    row = await db.execute(
        text("""
            SELECT COALESCE(MAX((metadata->>'turn_index')::int), -1) + 1
            FROM memories
            WHERE metadata->>'type' = 'chat' AND metadata->>'session_id' = :sid
        """),
        {"sid": session_id},
    )
    return row.scalar_one()


# Backward-compatible alias for API
async def hybrid_search(
    db: AsyncSession,
    query: str,
    emb: list[float],
    top_k: int = 5,
    filter_type: MemType | None = None,
    session_id: str | None = None,
) -> list[dict]:
    if filter_type == "doc" or filter_type is None:
        return await search_documents(db, query, emb, session_id or "", top_k=top_k)
    return await dense_search(db, emb, top_k, filter_type, session_id, "single")
