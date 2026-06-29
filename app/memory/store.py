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


async def _insert_memory(
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
    return r.scalar_one()


async def add_memory(
    db: AsyncSession,
    content: str,
    embedding: list[float],
    metadata: dict,
    chunk_level: ChunkLvl = "single",
    parent_id: int | None = None,
    *,
    commit: bool = True,
) -> int:
    mid = await _insert_memory(db, content, embedding, metadata, chunk_level, parent_id)
    if commit:
        await db.commit()
    return mid


async def add_parent_child(
    db: AsyncSession,
    parent_content: str,
    parent_emb: list[float],
    children: list[dict],
    metadata: dict,
    *,
    commit: bool = False,
) -> tuple[int, list[int]]:
    """Вставка parent + children; по умолчанию без commit (батч на уровне документа)."""
    pid = await _insert_memory(
        db, parent_content, parent_emb, {**metadata, "chunk_level": "parent"}, "parent"
    )
    cids: list[int] = []
    for c in children:
        child_meta = {**metadata, "chunk_level": "child", "chunk_index": c.get("index", 0)}
        if extra := c.get("metadata"):
            child_meta.update(extra)
        cid = await _insert_memory(
            db,
            c["content"],
            c["embedding"],
            child_meta,
            "child",
            pid,
        )
        cids.append(cid)
    if commit:
        await db.commit()
    return pid, cids


async def delete_document(
    db: AsyncSession,
    source: str,
    scope: DocScope,
    session_id: str | None = None,
    *,
    commit: bool = True,
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
    if commit:
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


async def keyword_search(
    db: AsyncSession,
    query: str,
    top_k: int = 20,
    session_id: str | None = None,
) -> list[dict]:
    """Полнотекстовый поиск по child-чанкам (tsvector, config simple)."""
    q = query.strip()
    if not q:
        return []
    conds = [
        "metadata->>'type' = 'doc'",
        "chunk_level = 'child'",
        "(metadata->>'scope' = 'global' OR "
        "(metadata->>'scope' = 'session' AND metadata->>'session_id' = :sid))",
        "tsvec @@ plainto_tsquery('simple', :q)",
    ]
    rows = await db.execute(
        text(f"""
        SELECT id, content, metadata, chunk_level, parent_id,
               ts_rank(tsvec, plainto_tsquery('simple', :q)) AS score
        FROM memories
        WHERE {" AND ".join(conds)}
        ORDER BY score DESC
        LIMIT :top_k
    """),
        {"q": q, "sid": session_id or "", "top_k": top_k},
    )
    return [dict(r._mapping) for r in rows.fetchall()]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Объединить несколько ранжированных списков hits по RRF."""
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            hid = hit["id"]
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank)
            items[hid] = hit
    ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    merged: list[dict] = []
    for hid in ordered:
        row = dict(items[hid])
        row["score"] = scores[hid]
        merged.append(row)
    return merged


def _heading_path_key(meta: dict) -> str:
    path = meta.get("heading_path")
    if isinstance(path, list):
        return " > ".join(str(p) for p in path)
    return ""


def _aggregate_parent_hits(hits: list[dict]) -> tuple[dict[int, float], dict[int, str]]:
    """child hits → parent scores с sibling bonus."""
    parent_score: dict[int, float] = {}
    parent_source: dict[int, str] = {}
    children_per_parent: dict[int, int] = {}

    for h in hits:
        pid = h.get("parent_id") or h["id"]
        sc = float(h.get("score") or 0.0)
        children_per_parent[pid] = children_per_parent.get(pid, 0) + 1
        if sc > parent_score.get(pid, 0.0):
            parent_score[pid] = sc
            parent_source[pid] = (h.get("metadata") or {}).get("source", "?")

    bonus = settings.retrieval_sibling_bonus
    for pid, count in children_per_parent.items():
        if count >= 2:
            parent_score[pid] = parent_score.get(pid, 0.0) + bonus * (count - 1)

    return parent_score, parent_source


async def search_documents(
    db: AsyncSession,
    query: str,
    emb: list[float],
    session_id: str,
    top_k: int = 5,
    *,
    with_scores: bool = False,
) -> list[dict] | tuple[list[dict], list[float], list[str]]:
    """Hybrid RAG: dense + keyword (RRF) → parent lift → sort → dedup."""
    candidate_k = top_k * settings.retrieval_candidate_multiplier

    dense_hits = await dense_search(
        db, emb, candidate_k, filter_type="doc", chunk_level="child",
        doc_scope_filter="documents", session_id=session_id,
    )

    if settings.retrieval_hybrid_enabled and query.strip():
        kw_hits = await keyword_search(db, query, candidate_k, session_id)
        hits = reciprocal_rank_fusion([dense_hits, kw_hits], k=settings.retrieval_rrf_k)
    else:
        hits = dense_hits

    parent_score, parent_source = _aggregate_parent_hits(hits)

    parents = await lift_to_parents(db, hits)
    parent_by_id = {p["id"]: p for p in parents}

    ranked_pids = sorted(
        parent_by_id.keys(),
        key=lambda pid: parent_score.get(pid, 0.0),
        reverse=True,
    )

    unique: list[dict] = []
    scores: list[float] = []
    sources: list[str] = []
    seen_paths: set[str] = set()

    for pid in ranked_pids:
        p = parent_by_id[pid]
        meta = p.get("metadata") or {}
        path_key = _heading_path_key(meta)
        if path_key and path_key in seen_paths:
            continue
        if path_key:
            seen_paths.add(path_key)
        unique.append(p)
        scores.append(parent_score.get(pid, 0.0))
        sources.append(parent_source.get(pid, meta.get("source", "?")))
        if len(unique) >= top_k:
            break

    if with_scores:
        return unique, scores, sources
    return unique


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
