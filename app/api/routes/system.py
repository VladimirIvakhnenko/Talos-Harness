import httpx
from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.database import get_db

router = APIRouter(tags=["System"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Статус всех компонентов")
async def health():
    checks: dict[str, str] = {}

    try:
        async with get_db() as db:
            row = await db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
            ver = row.scalar()
            checks["postgres_pgvector"] = f"ok (v{ver})" if ver else "pgvector not installed"
    except Exception as e:
        checks["postgres_pgvector"] = f"error: {e}"

    try:
        from app.tools.matiec_client import matiec_health

        mh = await matiec_health()
        checks["matiec"] = mh.get("status", "unknown")
    except Exception as e:
        checks["matiec"] = f"error: {e}"

    if settings.openrouter_api_key and settings.openrouter_api_key != "sk-or-v1-REPLACE_ME":
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                )
                checks["openrouter"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            checks["openrouter"] = f"error: {e}"
    else:
        checks["openrouter"] = "no api key"

    if settings.embedding_backend.lower() == "local":
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{settings.llama_embedding_url.rstrip('/')}/health")
                checks["llama_embedding"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            checks["llama_embedding"] = f"error: {e}"

    ok = all("ok" in v or v == "ok" for v in checks.values())
    return HealthResponse(status="ok" if ok else "degraded", checks=checks)
