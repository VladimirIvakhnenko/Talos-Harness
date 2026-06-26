from typing import Optional

from fastapi import APIRouter, Query

from app.database import get_db

router = APIRouter(tags=["Memory"])


@router.get("/memories/search", summary="Гибридный поиск по памяти (dense + sparse + RRF)")
async def memories_search(
    q: str = Query(..., description="Поисковый запрос"),
    type: Optional[str] = Query(None, description="doc | chat | fact | skill"),
    top_k: int = Query(5, ge=1, le=20),
):
    from app.agents.llm_client import embed_single
    from app.memory.store import hybrid_search

    emb = await embed_single(q)
    async with get_db() as db:
        results = await hybrid_search(db, q, emb, top_k=top_k, filter_type=type)
    return {"query": q, "results": results}
