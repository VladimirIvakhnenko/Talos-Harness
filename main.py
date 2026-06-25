"""
app/main.py — FastAPI приложение с полным Swagger UI.

Swagger: http://localhost:8000/docs
ReDoc:   http://localhost:8000/redoc
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db

settings = get_settings()
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_last_st: dict[str, str] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_db() as db:
        await db.execute(text("SELECT 1"))
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Talos Harness — Multi-Agent PLC Code Generator",
    description="""
## Многоагентная система генерации и верификации ST-кода для ПЛК

**Архитектура:** Planner (Nex-N2-Pro) → Engineer (Qwen3.5-9B) → Retriever (Qwen3.5-4B)  
**Провайдер LLM:** OpenRouter (все модели)  
**Верификация:** MatIEC (iec2c / iec2iec)  
**Память:** PostgreSQL 17 + pgvector (dense + sparse + RRF)  
**Бенчмарк:** Agents4PLC v2.0 (96 задач)

### Сценарии:
1. Генерация ST-модуля для контроллера Эльбрус по таблице сигналов
2. Написание скриптов IEC 61131-3 ST с RAG по загруженной документации
3. Оценка на Agents4PLC с метриками Compilation Rate и Pass@1
4. Мониторинг токенов и стоимости каждого вызова LLM
    """,
    version="2.0.0",
    contact={"name": "Talos Harness", "url": "https://github.com/talos-harness"},
    license_info={"name": "Apache 2.0"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., description="Запрос пользователя на естественном языке")
    session_id: Optional[str] = Field(None, description="UUID сессии; если не задан — создаётся автоматически")

    class Config:
        json_schema_extra = {"example": {"message": "Напиши функциональный блок для управления насосом с защитой от сухого хода", "session_id": None}}


class ChatResponse(BaseModel):
    session_id: str
    response: str
    final_code: Optional[str] = None
    matiec_ok: Optional[bool] = None
    steps: int = 0


class GenerateModuleRequest(BaseModel):
    controller: str = Field("elbrus", description="elbrus | baikal | codesys")
    signals_path: Optional[str] = Field(None, description="Путь к CSV с сигналами (после /upload_signals)")
    module_name: str = Field("GeneratedModule", description="Имя PROGRAM или FUNCTION_BLOCK")
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {"example": {"controller": "elbrus", "signals_path": "/app/uploads/signals_abc.csv", "module_name": "PumpController"}}


class GenerateModuleResponse(BaseModel):
    session_id: str
    controller: str
    module_name: str
    code: str
    matiec_ok: Optional[bool] = None
    matiec_errors: list[str] = []
    download_url: str


class ValidateRequest(BaseModel):
    code: str = Field(..., description="ST-код для верификации")
    task_id: str = Field("manual", description="Идентификатор задачи")

    class Config:
        json_schema_extra = {"example": {"code": "PROGRAM Test\nVAR x: BOOL; END_VAR\nx := TRUE;\nEND_PROGRAM", "task_id": "test_001"}}


class ValidateResponse(BaseModel):
    ok: bool
    compilation_rate: float
    errors: list[str]
    warnings: list[str]


class BenchmarkRunRequest(BaseModel):
    subset: str = Field("medium", description="easy | medium | hard")
    n_tasks: int = Field(10, ge=1, le=96, description="Количество задач для оценки")
    configs: list[str] = Field(["baseline", "full_agent"], description="Конфигурации для сравнения")

    class Config:
        json_schema_extra = {"example": {"subset": "medium", "n_tasks": 10, "configs": ["baseline", "full_agent"]}}


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str]


# ══════════════════════════════════════════════════════════════════════════════
# System
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Статус всех компонентов")
async def health():
    checks: dict[str, str] = {}

    # PostgreSQL + pgvector
    try:
        async with get_db() as db:
            row = await db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
            ver = row.scalar()
            checks["postgres_pgvector"] = f"ok (v{ver})" if ver else "pgvector not installed"
    except Exception as e:
        checks["postgres_pgvector"] = f"error: {e}"

    # MatIEC
    try:
        from app.tools.matiec_client import matiec_health
        mh = await matiec_health()
        checks["matiec"] = mh.get("status", "unknown")
    except Exception as e:
        checks["matiec"] = f"error: {e}"

    # OpenRouter
    if settings.openrouter_api_key and settings.openrouter_api_key != "sk-or-v1-REPLACE_ME":
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get("https://openrouter.ai/api/v1/models",
                                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"})
                checks["openrouter"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            checks["openrouter"] = f"error: {e}"
    else:
        checks["openrouter"] = "no api key"

    ok = all("ok" in v or "ok" == v for v in checks.values())
    return HealthResponse(status="ok" if ok else "degraded", checks=checks)


# ══════════════════════════════════════════════════════════════════════════════
# Documents
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload_pdf", tags=["Documents"],
          summary="Загрузка PDF документации (IEC 61131-3, Эльбрус, ТЗ)")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF файл"),
    doc_type: str = Query("general", description="iec_standard | elbrus_manual | tz | general"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())

    # Запуск обработки в фоне (Day 2 — полная реализация)
    return {
        "status":   "queued",
        "filename": file.filename,
        "doc_type": doc_type,
        "path":     str(path),
        "pipeline": "pdf2image → Qwen3.5-9B OCR → parent-child chunking → pgvector",
    }


@app.post("/upload_signals", tags=["Documents"],
          summary="Загрузка таблицы сигналов CSV/XLSX")
async def upload_signals(
    file: UploadFile = File(..., description="CSV или XLSX с таблицей сигналов"),
    controller: str = Query("elbrus", description="elbrus | baikal | codesys"),
):
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only CSV/XLSX accepted")
    path = UPLOAD_DIR / f"signals_{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())
    return {
        "status":     "saved",
        "filename":   file.filename,
        "controller": controller,
        "path":       str(path),
        "next":       f"POST /generate_module with signals_path={path}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse, tags=["Agent"],
          summary="ReAct-агент: генерация ST-кода на естественном языке")
async def chat(body: ChatRequest):
    from app.agents.react_agent import run_agent
    sid = body.session_id or str(uuid.uuid4())
    result = await run_agent(body.message, session_id=sid)
    return ChatResponse(**result)


@app.post("/generate_module", response_model=GenerateModuleResponse, tags=["Agent"],
          summary="Генерация ST-модуля для Эльбрус/Байкал по таблице сигналов")
async def generate_module(body: GenerateModuleRequest):
    from app.tools.signal_parser import parse_signal_table, signals_to_st_var, signals_to_scale_body
    from app.tools.matiec_client import compile_st

    sid = body.session_id or str(uuid.uuid4())

    if body.signals_path and Path(body.signals_path).exists():
        signals = parse_signal_table(body.signals_path)
        var_section  = signals_to_st_var(signals, body.controller)
        scale_body   = signals_to_scale_body(signals)
    else:
        var_section  = "VAR\n    (* No signals loaded — upload CSV via /upload_signals *)\nEND_VAR"
        scale_body   = "(* Add scaling logic here *)"

    code = f"""(* ================================================================ *)
(* MODULE: {body.module_name}                                         *)
(* CONTROLLER: {body.controller.upper()} (Elbrus-2C3)                *)
(* GENERATED BY: Talos Harness v2.0                                   *)
(* IEC 61131-3 Structured Text                                        *)
(* ================================================================ *)

PROGRAM {body.module_name}
{var_section}

{scale_body}

(* ── Control logic ─────────────────────────────────────────────── *)
(* Add your control logic here                                        *)

(* VERIFY: Check all AI scaling, DO outputs and alarm conditions *)
END_PROGRAM
"""
    # MatIEC валидация
    mat = await compile_st(code, sid)

    out_path = UPLOAD_DIR / f"module_{sid}.st"
    out_path.write_text(code, encoding="utf-8")
    _last_st[sid] = str(out_path)

    return GenerateModuleResponse(
        session_id=sid, controller=body.controller,
        module_name=body.module_name, code=code,
        matiec_ok=mat.ok, matiec_errors=mat.errors,
        download_url=f"/module/download?session_id={sid}",
    )


@app.get("/module/download", tags=["Agent"],
         summary="Скачать сгенерированный .st файл")
async def download_module(session_id: str = Query(...)):
    path = _last_st.get(session_id)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Module not found. Run /generate_module first.")
    return FileResponse(path, filename="module.st", media_type="text/plain")


@app.post("/validate", response_model=ValidateResponse, tags=["Agent"],
          summary="Валидация ST-кода через MatIEC (iec2c)")
async def validate_code(body: ValidateRequest):
    from app.tools.matiec_client import compile_st
    r = await compile_st(body.code, body.task_id)
    return ValidateResponse(
        ok=r.ok, compilation_rate=r.compilation_rate,
        errors=r.errors, warnings=r.warnings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/benchmark/run", tags=["Benchmark"],
          summary="Запуск оценки на Agents4PLC v2.0")
async def benchmark_run(body: BenchmarkRunRequest):
    from benchmark.runner import run_benchmark
    async with get_db() as db:
        result = await run_benchmark(
            subset=body.subset, n_tasks=body.n_tasks,
            configs=body.configs, db=db,
        )
    return result


@app.get("/benchmark/results", tags=["Benchmark"],
         summary="Результаты последних прогонов Agents4PLC")
async def benchmark_results():
    async with get_db() as db:
        rows = await db.execute(text("SELECT * FROM v_benchmark_summary LIMIT 50"))
        return {"results": [dict(r._mapping) for r in rows.fetchall()]}


# ══════════════════════════════════════════════════════════════════════════════
# Memory / Search
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/memories/search", tags=["Memory"],
         summary="Гибридный поиск по памяти (dense + sparse + RRF)")
async def memories_search(
    q:     str = Query(..., description="Поисковый запрос"),
    type:  Optional[str] = Query(None, description="doc | chat | fact | skill"),
    top_k: int = Query(5, ge=1, le=20),
):
    from app.agents.llm_client import embed_single
    from app.memory.store import hybrid_search
    emb = await embed_single(q)
    async with get_db() as db:
        results = await hybrid_search(db, q, emb, top_k=top_k, filter_type=type)
    return {"query": q, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
# Monitoring
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/monitoring/tokens", tags=["Monitoring"],
         summary="Суммарные токены по сессиям и агентам")
async def monitoring_tokens(session_id: Optional[str] = Query(None)):
    async with get_db() as db:
        if session_id:
            rows = await db.execute(
                text("SELECT * FROM v_token_summary WHERE session_id=:sid"),
                {"sid": session_id})
        else:
            rows = await db.execute(
                text("SELECT * FROM v_token_summary ORDER BY last_call_at DESC LIMIT 100"))
        return {"tokens": [dict(r._mapping) for r in rows.fetchall()]}


@app.get("/monitoring/cost", tags=["Monitoring"],
         summary="Стоимость вызовов в USD с разбивкой по агентам")
async def monitoring_cost(
    from_date: Optional[str] = Query(None, alias="from"),
    to_date:   Optional[str] = Query(None, alias="to"),
):
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT agent_name, model_id,
                   COUNT(*) AS calls,
                   SUM(total_tokens) AS total_tokens,
                   ROUND(SUM(cost_usd)::NUMERIC,6) AS total_cost_usd,
                   AVG(latency_ms)::INT AS avg_latency_ms
            FROM token_usage
            WHERE (:from_d IS NULL OR created_at >= :from_d::TIMESTAMPTZ)
              AND (:to_d   IS NULL OR created_at <= :to_d::TIMESTAMPTZ)
            GROUP BY agent_name, model_id
            ORDER BY total_cost_usd DESC
        """), {"from_d": from_date, "to_d": to_date})
        return {"cost_breakdown": [dict(r._mapping) for r in rows.fetchall()]}


@app.get("/monitoring/cost/per_task", tags=["Monitoring"],
         summary="Средняя стоимость задачи Agents4PLC по сложности и конфигурации")
async def cost_per_task():
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT b.config, b.difficulty,
                   COUNT(*) AS tasks,
                   ROUND(AVG(t.cost_usd)::NUMERIC*1000,4) AS avg_cost_per_task_milli_usd
            FROM benchmark_results b
            LEFT JOIN token_usage t ON t.task_id = b.task_id
            GROUP BY b.config, b.difficulty
            ORDER BY b.config, b.difficulty
        """))
        return {"per_task": [dict(r._mapping) for r in rows.fetchall()]}


@app.get("/monitoring/dashboard", response_class=HTMLResponse, tags=["Monitoring"],
         summary="Plotly HTML-дашборд: токены / стоимость / latency")
async def monitoring_dashboard():
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT agent_name, model_id,
                   SUM(total_tokens) AS tokens,
                   ROUND(SUM(cost_usd)::NUMERIC,6) AS cost,
                   AVG(latency_ms)::INT AS avg_lat
            FROM token_usage GROUP BY agent_name, model_id ORDER BY tokens DESC
        """))
        data = [dict(r._mapping) for r in rows.fetchall()]

    agents  = [d["agent_name"] or "unknown" for d in data]
    tokens  = [int(d["tokens"] or 0)         for d in data]
    costs   = [float(d["cost"] or 0)         for d in data]
    latency = [int(d["avg_lat"] or 0)        for d in data]

    html = f"""<!DOCTYPE html><html>
<head><title>Talos Harness — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:Arial,sans-serif;background:#0f1117;color:#eee;padding:20px;margin:0}}
h1{{color:#4BACC6;}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}}
.card{{background:#1e2130;border-radius:12px;padding:16px}}</style></head>
<body>
<h1>🤖 Talos Harness — Token & Cost Dashboard</h1>
<p>Модели: Planner=<b>nex-agi/nex-n2-pro</b> · Engineer=<b>qwen/qwen3.5-9b</b> · Retriever=<b>qwen/qwen3.5-4b</b></p>
<div class="grid">
  <div class="card"><div id="t"></div></div>
  <div class="card"><div id="c"></div></div>
  <div class="card"><div id="l"></div></div>
</div>
<script>
const a={agents}; const t={tokens}; const c={costs}; const l={latency};
const bg='#1e2130'; const fc={{color:'#eee'}};
Plotly.newPlot('t',[{{type:'bar',x:a,y:t,marker:{{color:'#4BACC6'}}}}],
  {{title:'Tokens by Agent',paper_bgcolor:bg,plot_bgcolor:bg,font:fc}});
Plotly.newPlot('c',[{{type:'bar',x:a,y:c,marker:{{color:'#2E75B6'}}}}],
  {{title:'Cost USD by Agent',paper_bgcolor:bg,plot_bgcolor:bg,font:fc}});
Plotly.newPlot('l',[{{type:'bar',x:a,y:l,marker:{{color:'#375623'}}}}],
  {{title:'Avg Latency ms',paper_bgcolor:bg,plot_bgcolor:bg,font:fc}});
</script></body></html>"""
    return HTMLResponse(html)


# ══════════════════════════════════════════════════════════════════════════════
# __init__ stubs
# ══════════════════════════════════════════════════════════════════════════════
# Пустые файлы __init__.py создаются в setup