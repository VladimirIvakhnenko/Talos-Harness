from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.database import get_db

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class SessionCreate(BaseModel):
    title: str = Field("Новый чат", max_length=120)


@router.get("", summary="Список чатов")
async def list_sessions(limit: int = 50):
    from app.memory.sessions import list_sessions as _list

    async with get_db() as db:
        return {"sessions": await _list(db, limit=limit)}


@router.post("", summary="Создать чат")
async def create_session(body: SessionCreate):
    from app.memory.sessions import create_session as _create

    async with get_db() as db:
        sid = await _create(db, body.title)
    return {"session_id": sid, "title": body.title}


@router.get("/{session_id}/messages", summary="История сообщений чата")
async def get_messages(session_id: str):
    from app.memory.store import load_chat_history

    async with get_db() as db:
        messages = await load_chat_history(db, session_id)
    return {"session_id": session_id, "messages": messages}


@router.patch("/{session_id}", summary="Переименовать чат")
async def rename_session(session_id: str, body: SessionCreate):
    from app.memory.sessions import rename_session as _rename

    async with get_db() as db:
        await _rename(db, session_id, body.title)
    return {"session_id": session_id, "title": body.title}
