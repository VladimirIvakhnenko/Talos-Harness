"""
app/memory/store.py — Векторная память: dense + sparse + RRF + parent-child.
"""
from __future__ import annotations
import json
from typing import Any, Literal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

settings = get_settings()

MemType  = Literal["doc", "chat", "fact", "skill"]
ChunkLvl = Literal["child", "parent", "single"]


def _embedding_expr(column: str = "embedding") -> str:
    """pgvector HNSW indexes support up to 2000 dims; use halfvec cast above that."""
    dims = settings.embedding_dimensions
    if dims > 2000:
        return f"({column}::halfvec({dims}))"
    return column


def _query_embedding_cast() -> str:
    dims = settings.embedding_dimensions
    if dims > 2000:
        return f"CAST(:emb AS halfvec({dims}))"
    return "CAST(:emb AS vector)"


async def add_memory(db: AsyncSession, content: str, embedding: list[float],
                     metadata: dict, chunk_level: ChunkLvl = "single",
                     parent_id: int | None = None) -> int:
    r = await db.execute(text("""
        INSERT INTO memories (content, embedding, metadata, chunk_level, parent_id)
        VALUES (:c, CAST(:e AS vector), CAST(:m AS jsonb), :l, :p) RETURNING id
    """), {
        "c": content,
        "e": str(embedding),
        "m": json.dumps(metadata),
        "l": chunk_level,
        "p": parent_id,
    })
    await db.commit()
    return r.scalar_one()


async def add_parent_child(db: AsyncSession, parent_content: str,
                           parent_emb: list[float], children: list[dict],
                           metadata: dict) -> tuple[int, list[int]]:
    pid = await add_memory(db, parent_content, parent_emb,
                           {**metadata, "chunk_level": "parent"}, "parent")
    cids = []
    for c in children:
        cid = await add_memory(db, c["content"], c["embedding"],
                               {**metadata, "chunk_level": "child",
                                "chunk_index": c.get("index", 0)},
                               "child", pid)
        cids.append(cid)
    return pid, cids


async def dense_search(db: AsyncSession, emb: list[float], top_k: int = 20,
                       filter_type: MemType | None = None,
                       session_id: str | None = None,
                       chunk_level: ChunkLvl | None = "child") -> list[dict]:
    conds, params = ["1=1"], {"emb": str(emb), "top_k": top_k}
    if filter_type:
        conds.append("metadata->>'type' = :ft"); params["ft"] = filter_type
    if session_id:
        conds.append("metadata->>'session_id' = :sid"); params["sid"] = session_id
    if chunk_level:
        conds.append("chunk_level = :lvl"); params["lvl"] = chunk_level
    emb_expr = _embedding_expr()
    emb_cast = _query_embedding_cast()
    rows = await db.execute(text(f"""
        SELECT id,content,metadata,chunk_level,parent_id,
               1-({emb_expr} <=> {emb_cast}) AS score
        FROM memories WHERE {" AND ".join(conds)}
        ORDER BY {emb_expr} <=> {emb_cast} LIMIT :top_k
    """), params)
    return [dict(r._mapping) for r in rows.fetchall()]


async def sparse_search(db: AsyncSession, query: str, top_k: int = 20,
                        filter_type: MemType | None = None,
                        chunk_level: ChunkLvl | None = "child") -> list[dict]:
    conds = ["tsvec @@ plainto_tsquery('russian',:q)"]
    params: dict[str, Any] = {"q": query, "top_k": top_k}
    if filter_type:
        conds.append("metadata->>'type'=:ft"); params["ft"] = filter_type
    if chunk_level:
        conds.append("chunk_level=:lvl"); params["lvl"] = chunk_level
    rows = await db.execute(text(f"""
        SELECT id,content,metadata,chunk_level,parent_id,
               ts_rank(tsvec,plainto_tsquery('russian',:q)) AS score
        FROM memories WHERE {" AND ".join(conds)}
        ORDER BY score DESC LIMIT :top_k
    """), params)
    return [dict(r._mapping) for r in rows.fetchall()]


def rrf(dense: list[dict], sparse: list[dict], k: int = 60,
        top_k: int = 20) -> list[dict]:
    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}
    for rank, d in enumerate(dense, 1):
        scores[d["id"]] = scores.get(d["id"], 0) + 1/(k+rank)
        docs[d["id"]] = d
    for rank, d in enumerate(sparse, 1):
        scores[d["id"]] = scores.get(d["id"], 0) + 1/(k+rank)
        docs[d["id"]] = d
    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [{**docs[i], "rrf_score": scores[i]} for i in sorted_ids[:top_k]]


async def lift_to_parents(db: AsyncSession, results: list[dict]) -> list[dict]:
    pids = list({d["parent_id"] for d in results if d.get("parent_id")})
    singles = [d for d in results if not d.get("parent_id")]
    if not pids:
        return singles
    rows = await db.execute(
        text("SELECT id,content,metadata,chunk_level,parent_id FROM memories WHERE id=ANY(:ids)"),
        {"ids": pids})
    return [dict(r._mapping) for r in rows.fetchall()] + singles


async def hybrid_search(db: AsyncSession, query: str, emb: list[float],
                        top_k: int = 5, filter_type: MemType | None = None,
                        session_id: str | None = None) -> list[dict]:
    dn = await dense_search(db, emb, 20, filter_type, session_id)
    sp = await sparse_search(db, query, 20, filter_type)
    fused = rrf(dn, sp, top_k=20)
    parents = await lift_to_parents(db, fused)
    return parents[:top_k]


async def save_chat_turn(db: AsyncSession, user_msg: str, assistant_msg: str,
                         emb: list[float], session_id: str, tokens: int = 0):
    await add_memory(db, f"User: {user_msg}\nAssistant: {assistant_msg}", emb,
                     {"type": "chat", "session_id": session_id, "tokens": tokens})


async def save_fact(db: AsyncSession, fact: str, emb: list[float], session_id: str):
    await add_memory(db, fact, emb, {"type": "fact", "session_id": session_id})


async def get_recent_messages(db: AsyncSession, session_id: str,
                              limit: int = 10) -> list[dict]:
    rows = await db.execute(text("""
        SELECT content,metadata,created_at FROM memories
        WHERE metadata->>'type'='chat' AND metadata->>'session_id'=:sid
        ORDER BY created_at DESC LIMIT :lim
    """), {"sid": session_id, "lim": limit})
    return list(reversed([dict(r._mapping) for r in rows.fetchall()]))


async def get_chat_context(db: AsyncSession, query: str, emb: list[float],
                           session_id: str) -> str:
    recent   = await get_recent_messages(db, session_id, 10)
    semantic = await dense_search(db, emb, 3, "chat", session_id, "single")
    facts    = await dense_search(db, emb, 2, "fact", session_id, "single")
    parts = []
    if facts:
        parts.append("[Known facts]\n" + "\n".join(f["content"] for f in facts))
    if semantic:
        parts.append("[Relevant history]\n" + "\n---\n".join(s["content"] for s in semantic))
    if recent:
        parts.append("[Recent messages]\n" + "\n".join(m["content"] for m in recent))
    return "\n\n".join(parts)