from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import BenchmarkRunRequest, StCodingBenchRunRequest
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


@router.post("/benchmark/st_coding/run", summary="Прогон ST coding benchmark — одна config за вызов")
async def st_coding_benchmark_run(body: StCodingBenchRunRequest):
    from benchmark.st_coding_runner import run_st_coding_benchmark

    async with get_db() as db:
        result = await run_st_coding_benchmark(
            n_tasks=body.n_tasks,
            config=body.config,
            configs=body.configs,
            guide_path=body.guide_path,
            max_validation_attempts=body.max_validation_attempts,
            run_id=body.run_id,
            start_task_id=body.start_task_id,
            db=db,
        )
    return result
