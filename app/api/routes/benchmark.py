from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import BenchmarkRunRequest
from app.database import get_db

router = APIRouter(tags=["Benchmark"])


@router.post("/benchmark/run", summary="Запуск оценки на Agents4PLC v2.0")
async def benchmark_run(body: BenchmarkRunRequest):
    from benchmark.runner import run_benchmark

    async with get_db() as db:
        result = await run_benchmark(
            subset=body.subset,
            n_tasks=body.n_tasks,
            configs=body.configs,
            db=db,
        )
    return result


@router.get("/benchmark/results", summary="Результаты последних прогонов Agents4PLC")
async def benchmark_results():
    async with get_db() as db:
        rows = await db.execute(text("SELECT * FROM v_benchmark_summary LIMIT 50"))
        return {"results": [dict(r._mapping) for r in rows.fetchall()]}
