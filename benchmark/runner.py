"""
benchmark/runner.py — Запуск оценки на Agents4PLC v2.0.

Читает задачи из benchmark_v2/medium.jsonl (или easy/hard),
запускает агента, верифицирует через MatIEC, записывает в benchmark_results.
"""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path

from app.agents.react_agent import run_agent
from app.tools.matiec_client import compile_st
from app.config import get_settings

settings = get_settings()

BENCH_DIR = Path(__file__).parent / "Agents4PLC_release" / "benchmark_v2"


async def run_benchmark(
    subset: str = "medium",
    n_tasks: int = 10,
    configs: list[str] | None = None,
    db=None,
) -> dict:
    """
    configs: ["baseline", "rag_only", "full_agent"]
    """
    configs = configs or ["baseline", "full_agent"]
    run_id = str(uuid.uuid4())

    jsonl_path = BENCH_DIR / f"{subset}.jsonl"
    if not jsonl_path.exists():
        return {"error": f"Benchmark file not found: {jsonl_path}",
                "hint": "Clone https://github.com/Luoji-zju/Agents4PLC_release into benchmark/"}

    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    tasks = tasks[:n_tasks]

    all_results = []
    for config in configs:
        for task in tasks:
            result = await _run_single(run_id, task, config, subset, db)
            all_results.append(result)

    summary = _compute_summary(all_results)
    return {"run_id": run_id, "summary": summary, "results": all_results}


async def _run_single(run_id: str, task: dict, config: str,
                      difficulty: str, db) -> dict:
    task_id    = task.get("task_id", "?")
    desc       = task.get("description", "")
    formal_spec = task.get("formal_spec", "")

    t0 = time.perf_counter()

    if config == "baseline":
        # Только LLM, без RAG
        from app.agents.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.agents.react_agent import _skill_registry
        from app.skills.prompt_builder import build_engineer_prompt

        active = _skill_registry.active_slugs if _skill_registry else None
        prompt_text = build_engineer_prompt(ENGINEER_PROMPT, _skill_registry, active)
        llm = get_llm("engineer", task_id=task_id, tool_name="baseline_gen")
        from app.prompts.system_prompts import BENCHMARK_PROMPT
        prompt = BENCHMARK_PROMPT.format(description=desc, formal_spec=formal_spec)
        resp = await llm.ainvoke([SystemMessage(content=prompt_text),
                                   HumanMessage(content=prompt)])
        code = resp.content
    else:
        # Full agent с ReAct + RAG
        from app.prompts.system_prompts import BENCHMARK_PROMPT
        prompt = BENCHMARK_PROMPT.format(description=desc, formal_spec=formal_spec)
        agent_result = await run_agent(prompt, task_id=task_id)
        code = agent_result.get("final_code") or agent_result.get("response", "")

    latency = int((time.perf_counter() - t0) * 1000)

    # MatIEC компиляция
    compile_result = await compile_st(code or "", task_id)

    row = {
        "run_id":                 run_id,
        "task_id":                task_id,
        "difficulty":             difficulty,
        "config":                 config,
        "generated_code":         code,
        "compilation_ok":         compile_result.ok,
        "formal_verification_ok": None,  # nuXmv — опционально
        "execution_correct":      None,
        "matiec_errors":          compile_result.errors,
        "latency_ms":             latency,
    }

    if db:
        from sqlalchemy import text
        await db.execute(text("""
            INSERT INTO benchmark_results
              (run_id,task_id,difficulty,config,generated_code,
               compilation_ok,formal_verification_ok,execution_correct,
               matiec_errors,latency_ms)
            VALUES (:run_id,:task_id,:difficulty,:config,:code,
                    :comp,:fv,:exec,:merr,:lat)
        """), {
            "run_id": run_id, "task_id": task_id, "difficulty": difficulty,
            "config": config, "code": code,
            "comp": compile_result.ok, "fv": None, "exec": None,
            "merr": json.dumps(compile_result.errors), "lat": latency,
        })
        await db.commit()

    return row


def _compute_summary(results: list[dict]) -> dict:
    by_config: dict[str, dict] = {}
    for r in results:
        c = r["config"]
        if c not in by_config:
            by_config[c] = {"total": 0, "compiled": 0}
        by_config[c]["total"] += 1
        if r.get("compilation_ok"):
            by_config[c]["compiled"] += 1

    summary = {}
    for c, v in by_config.items():
        total = v["total"]
        summary[c] = {
            "total_tasks":        total,
            "compilation_rate":   round(v["compiled"] / total * 100, 1) if total else 0,
        }
    return summary