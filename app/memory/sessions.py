"""
app/memory/sessions.py — управление сессиями чатов (таблица sessions).
"""
from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def create_session(db: AsyncSession, title: str = "Новый чат") -> str:
    sid = str(uuid.uuid4())
    await db.execute(
        text("INSERT INTO sessions (id, title) VALUES (CAST(:id AS uuid), :title)"),
        {"id": sid, "title": title[:120]},
    )
    await db.commit()
    return sid


async def list_sessions(db: AsyncSession, limit: int = 50) -> list[dict]:
    rows = await db.execute(
        text("""
            SELECT id::text AS id, title, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT :lim
        """),
        {"lim": limit},
    )
    return [dict(r._mapping) for r in rows.fetchall()]


async def touch_session(db: AsyncSession, session_id: str) -> None:
    await db.execute(
        text("UPDATE sessions SET updated_at = NOW() WHERE id = CAST(:id AS uuid)"),
        {"id": session_id},
    )
    await db.commit()


async def rename_session(db: AsyncSession, session_id: str, title: str) -> None:
    await db.execute(
        text("UPDATE sessions SET title = :title, updated_at = NOW() WHERE id = CAST(:id AS uuid)"),
        {"id": session_id, "title": title[:120]},
    )
    await db.commit()


async def ensure_session(db: AsyncSession, session_id: str | None, title: str = "Новый чат") -> str:
    """Вернуть существующий session_id или создать новую сессию."""
    if session_id:
        row = await db.execute(
            text("SELECT id::text FROM sessions WHERE id = CAST(:id AS uuid)"),
            {"id": session_id},
        )
        if row.scalar_one_or_none():
            return session_id
    return await create_session(db, title)
