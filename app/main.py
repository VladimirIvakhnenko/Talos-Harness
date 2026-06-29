"""
app/main.py — FastAPI приложение с полным Swagger UI.

Swagger: http://localhost:8000/docs
ReDoc:   http://localhost:8000/redoc
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.api.routes import agent, benchmark, documents, memory, monitoring, sessions, system, skills
from app.config import get_settings
from app.database import get_db

APP_DESCRIPTION = """
## Многоагентная система генерации и верификации ST-кода для ПЛК

**Архитектура:** Planner (Nex-N2-Pro) → Engineer (Qwen3.5-9B) → Retriever (Qwen3.5-4B)  
**Провайдер LLM:** OpenRouter (все модели)  
**Верификация:** MatIEC (iec2c / iec2iec)  
**Память:** PostgreSQL 17 + pgvector (dense RAG, сессии чатов)  
**Бенчмарк:** Agents4PLC v2.0 (96 задач)

### Сценарии:
1. Генерация ST-модуля для контроллера Эльбрус по таблице сигналов
2. Написание скриптов IEC 61131-3 ST с RAG по загруженной документации
3. Оценка на Agents4PLC с метриками Compilation Rate и Pass@1
4. Мониторинг токенов и стоимости каждого вызова LLM
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # DB health check
    async with get_db() as db:
        await db.execute(text("SELECT 1"))

    # Init skill registry
    if settings.skills_enabled:
        from app.agents.react_agent import set_skill_registry as set_agent_registry
        from app.skills.registry import SkillRegistry

        registry = SkillRegistry()
        discovered = registry.discover_all()
        app.state.skill_registry = registry
        app.state.skills_enabled = True
        set_agent_registry(registry)
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Skill registry initialized: %d skills discovered", len(discovered))
    else:
        app.state.skills_enabled = False

    yield


app = FastAPI(
    title="Talos Harness — Multi-Agent PLC Code Generator",
    description=APP_DESCRIPTION,
    version="2.0.0",
    contact={"name": "Talos Harness", "url": "https://github.com/talos-harness"},
    license_info={"name": "Apache 2.0"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(agent.router)
app.include_router(benchmark.router)
app.include_router(memory.router)
app.include_router(sessions.router)
app.include_router(monitoring.router)
app.include_router(skills.router)
