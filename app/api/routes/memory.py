from fastapi import APIRouter, Query

from app.database import get_db

router = APIRouter(tags=["Memory"])


@router.get("/memories/search", summary="Поиск по документации (dense, global + session)")
async def memories_search(
    q: str = Query(..., description="Поисковый запрос"),
    session_id: str = Query("", description="UUID чата для session-scoped docs"),
    top_k: int = Query(5, ge=1, le=20),
):
    from app.agents.llm_client import embed_single
    from app.memory.store import search_documents

    emb = await embed_single(q)
    async with get_db() as db:
        results = await search_documents(db, q, emb, session_id, top_k=top_k)
    return {"query": q, "session_id": session_id, "results": results}
