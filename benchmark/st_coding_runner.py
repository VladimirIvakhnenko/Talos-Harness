"""
benchmark/st_coding_runner.py — Прогон ST coding benchmark (st_coding_bench.json).

Configs:
  vanilla_llm       — LLM-only без RAG, без скиллов (нижний бейзлайн)
  rag_only          — ReAct + RAG (векторный поиск), без скиллов
  rag_skills        — ReAct + RAG + все builtin скиллы принудительно
  rag_skill_router  — ReAct + RAG + авто-роутинг скиллов (флагман)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from app.agents.llm_client import embed_single, get_llm
from app.agents.react_agent import extract_user_facing_response, run_agent
from app.config import get_settings
from app.memory.sessions import ensure_session
from app.tools.matiec_client import compile_st

settings = get_settings()

BENCH_FILE = Path(__file__).parent / "st_coding_bench.json"
DEFAULT_GUIDE = Path(__file__).parent / "assets" / "IEC-61131-3-ST-GUIDE.md"
VALID_CONFIGS = {"vanilla_llm", "rag_only", "rag_skills", "rag_skill_router"}
ALL_CONFIGS = ["vanilla_llm", "rag_only", "rag_skills", "rag_skill_router"]


def resolve_configs(
    config: str | None = None,
    configs: list[str] | None = None,
) -> list[str]:
    """Один прогон = одна config. configs[] оставлен для обратной совместимости."""
    if configs:
        chosen = configs
    elif config:
        chosen = [config]
    else:
        chosen = ["vanilla_llm"]
    unknown = set(chosen) - VALID_CONFIGS
    if unknown:
        raise ValueError(f"Unknown config(s): {sorted(unknown)}. Valid: {sorted(VALID_CONFIGS)}")
    return chosen


def load_tasks(n_tasks: int | None = None) -> list[dict]:
    raw = BENCH_FILE.read_text(encoding="utf-8-sig")
    tasks = json.loads(raw)
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    return tasks


def filter_tasks(
    tasks: list[dict],
    *,
    start_task_id: str | None = None,
    skip_task_ids: set[str] | None = None,
) -> list[dict]:
    filtered = list(tasks)
    if start_task_id:
        ids = [t["task_id"] for t in filtered]
        if start_task_id not in ids:
            raise ValueError(f"Unknown start_task_id {start_task_id!r}. Expected one of: {ids}")
        filtered = filtered[ids.index(start_task_id):]
    if skip_task_ids:
        filtered = [t for t in filtered if t["task_id"] not in skip_task_ids]
    return filtered


async def get_resume_context(
    db,
    config: str,
    session_id: str | None = None,
) -> dict | None:
    """Последний прогон с session_id для config — для продолжения в том же чате."""
    if session_id:
        row = await db.execute(
            text("""
                SELECT run_id::text AS run_id,
                       session_id::text AS session_id,
                       array_agg(DISTINCT task_id) AS completed
                FROM benchmark_results
                WHERE config = :cfg
                  AND benchmark_suite = 'st_coding'
                  AND session_id::text = :sid
                GROUP BY run_id, session_id
                ORDER BY MAX(created_at) DESC
                LIMIT 1
            """),
            {"cfg": config, "sid": session_id},
        )
    else:
        row = await db.execute(
            text("""
                SELECT run_id::text AS run_id,
                       session_id::text AS session_id,
                       array_agg(DISTINCT task_id) AS completed
                FROM benchmark_results
                WHERE config = :cfg
                  AND benchmark_suite = 'st_coding'
                  AND session_id IS NOT NULL
                GROUP BY run_id, session_id
                ORDER BY MAX(created_at) DESC
                LIMIT 1
            """),
            {"cfg": config},
        )
    r = row.mappings().first()
    if not r or not r["session_id"]:
        return None
    completed = r["completed"] or []
    return {
        "run_id": r["run_id"],
        "session_id": r["session_id"],
        "completed_task_ids": set(completed),
    }


async def _session_has_guide(db, session_id: str, source_name: str) -> bool:
    row = await db.execute(
        text("""
            SELECT 1 FROM memories
            WHERE metadata->>'type' = 'doc'
              AND metadata->>'scope' = 'session'
              AND metadata->>'session_id' = :sid
              AND metadata->>'source' = :src
            LIMIT 1
        """),
        {"sid": session_id, "src": source_name},
    )
    return row.scalar_one_or_none() is not None


def resolve_guide_path(guide_path: str | None) -> Path:
    if guide_path:
        p = Path(guide_path)
        if p.is_file():
            return p
    for candidate in (
        Path(settings.benchmark_st_guide_path),
        DEFAULT_GUIDE,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"ST guide not found (tried {guide_path!r}, {settings.benchmark_st_guide_path}, {DEFAULT_GUIDE})"
    )


async def _index_guide(
    db,
    guide_path: Path,
    *,
    session_id: str | None,
    scope: str,
) -> dict:
    from app.tools.text_processor import process_text_file

    return await process_text_file(
        str(guide_path),
        doc_type="iec_standard",
        source_name=guide_path.name,
        db=db,
        session_id=session_id,
        scope=scope,
    )


async def _aggregate_tokens(db, session_id: str, task_id: str) -> dict:
    # Flush pending token rows first
    from app.monitoring.token_tracker import flush_pending_tokens
    await flush_pending_tokens(db, session_id, task_id)

    row = await db.execute(
        text("""
            SELECT
                COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0)      AS total_tokens,
                COALESCE(SUM(cost_usd), 0)          AS cost_usd
            FROM token_usage
            WHERE session_id = :sid AND task_id = :tid
        """),
        {"sid": session_id, "tid": task_id},
    )
    m = row.mappings().first()
    return dict(m) if m else {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0,
    }


def _rag_metrics(retrieval_context: dict, guide_name: str) -> dict:
    scores = retrieval_context.get("retrieval_scores") or []
    sources = retrieval_context.get("retrieval_sources") or []
    guide_hit = any(guide_name.lower() in (s or "").lower() for s in sources)
    top1 = float(scores[0]) if scores else 0.0
    avg = sum(float(s) for s in scores) / len(scores) if scores else 0.0
    return {
        "retrieval_doc_count": retrieval_context.get("doc_count", 0),
        "retrieval_top1_score": round(top1, 4),
        "retrieval_avg_score": round(avg, 4),
        "guide_hit": guide_hit,
        "retrieval_sources": sources,
    }


async def _save_chat_turn(db, session_id: str, user_text: str, assistant_text: str) -> None:
    from app.memory.store import next_turn_index, save_chat_message

    emb_u = await embed_single(user_text[:2000])
    emb_a = await embed_single(assistant_text[:2000])
    ti = await next_turn_index(db, session_id)
    await save_chat_message(db, session_id, "user", user_text, emb_u, ti)
    await save_chat_message(db, session_id, "assistant", assistant_text, emb_a, ti + 1)


async def _run_vanilla_llm(task: dict, session_id: str) -> tuple[str, dict]:
    """Чистый LLM вызов — без RAG, без скиллов, без engineer prompt."""
    llm = get_llm(
        "engineer",
        session_id=session_id,
        task_id=task["task_id"],
        tool_name="vanilla_llm",
    )
    prompt = (
        f"Реализуй IEC 61131-3 Structured Text по спецификации.\n\n"
        f"{task['prompt']}\n\n"
        "Выведи только ST-код без пояснений."
    )
    resp = await llm.ainvoke([
        SystemMessage(content="Ты — эксперт IEC 61131-3. Сгенерируй ST-код."),
        HumanMessage(content=prompt),
    ])
    code = resp.content if isinstance(resp.content, str) else str(resp.content)
    return code, {}


async def _run_agent_task(
    task: dict,
    session_id: str,
    max_validation_attempts: int,
    skills: list[str] | None = None,
    route_skills: bool = False,
) -> tuple[str | None, dict]:
    agent_result = await run_agent(
        task["prompt"],
        session_id=session_id,
        task_id=task["task_id"],
        max_validation_attempts=max_validation_attempts,
        skills=skills,
        route_skills=route_skills,
    )
    code = agent_result.get("final_code")
    return code, agent_result


async def _run_single(
    run_id: str,
    task: dict,
    config: str,
    *,
    session_id: str,
    max_validation_attempts: int,
    guide_name: str,
    db,
    persist_chat: bool,
    skills: list[str] | None = None,
    route_skills: bool = False,
) -> dict:
    task_id = task["task_id"]
    t0 = time.perf_counter()

    if config == "vanilla_llm":
        code, agent_meta = await _run_vanilla_llm(task, session_id)
        resolved_task_id = task_id
    else:
        code, agent_meta = await _run_agent_task(
            task, session_id, max_validation_attempts,
            skills=skills, route_skills=route_skills,
        )
        resolved_task_id = agent_meta.get("task_id") or task_id

    latency_ms = int((time.perf_counter() - t0) * 1000)
    token_row = await _aggregate_tokens(db, session_id, resolved_task_id)

    compile_result = await compile_st(code or "", resolved_task_id)
    compilation_ok = compile_result.ok

    retrieval_ctx = agent_meta.get("retrieval_context") or {}
    rag = _rag_metrics(retrieval_ctx, guide_name) if config != "vanilla_llm" else {}

    validation_attempts = agent_meta.get("validation_attempts", 0)
    generate_attempts = agent_meta.get("generate_attempts", 0)
    pass_at_1 = bool(agent_meta.get("pass_at_1"))
    matiec_ok = agent_meta.get("matiec_ok")

    extra_metrics = {
        **rag,
        "tool_rounds": agent_meta.get("tool_rounds", 0),
        "matiec_ok_agent": matiec_ok,
        "prompt_tokens": int(token_row["prompt_tokens"]),
        "completion_tokens": int(token_row["completion_tokens"]),
        "failed_after_budget": (
            validation_attempts >= max_validation_attempts and not compilation_ok
        ),
    }

    if persist_chat and config != "vanilla_llm" and db:
        assistant_text = extract_user_facing_response([], agent_meta)
        await _save_chat_turn(db, session_id, task["prompt"], assistant_text)

    row = {
        "run_id": run_id,
        "task_id": task_id,
        "difficulty": "st_coding",
        "config": config,
        "generated_code": code,
        "compilation_ok": compilation_ok,
        "formal_verification_ok": None,
        "execution_correct": None,
        "matiec_errors": compile_result.errors,
        "total_tokens": int(token_row["total_tokens"]),
        "cost_usd": float(token_row["cost_usd"]),
        "latency_ms": latency_ms,
        "session_id": session_id,
        "validation_attempts": validation_attempts,
        "generate_attempts": generate_attempts,
        "pass_at_1": pass_at_1 and compilation_ok,
        "benchmark_suite": "st_coding",
        "extra_metrics": extra_metrics,
    }

    await db.execute(
        text("""
            INSERT INTO benchmark_results
              (run_id, task_id, difficulty, config, generated_code,
               compilation_ok, formal_verification_ok, execution_correct,
               matiec_errors, total_tokens, cost_usd, latency_ms,
               session_id, validation_attempts, generate_attempts,
               pass_at_1, benchmark_suite, extra_metrics)
            VALUES
              (:run_id, :task_id, :difficulty, :config, :code,
               :comp, :fv, :exec, CAST(:merr AS jsonb),
               :tokens, :cost, :lat,
               CAST(:sid AS uuid), :va, :ga,
               :p1, :suite, CAST(:extra AS jsonb))
        """),
        {
            "run_id": run_id,
            "task_id": task_id,
            "difficulty": "st_coding",
            "config": config,
            "code": code,
            "comp": compilation_ok,
            "fv": None,
            "exec": None,
            "merr": json.dumps(compile_result.errors),
            "tokens": row["total_tokens"],
            "cost": row["cost_usd"],
            "lat": latency_ms,
            "sid": session_id,
            "va": validation_attempts,
            "ga": generate_attempts,
            "p1": row["pass_at_1"],
            "suite": "st_coding",
            "extra": json.dumps(extra_metrics),
        },
    )
    await db.commit()
    return row


def _compute_summary(results: list[dict]) -> dict:
    by_config: dict[str, dict] = {}
    for r in results:
        cfg = r["config"]
        if cfg not in by_config:
            by_config[cfg] = {
                "total": 0,
                "compiled": 0,
                "pass_at_1": 0,
                "failed_first_recovered": 0,
                "failed_first_total": 0,
                "latency_sum": 0,
                "tokens_sum": 0,
                "cost_sum": 0.0,
                "validation_sum": 0,
                "guide_hits": 0,
                "top1_sum": 0.0,
                "top1_count": 0,
            }
        b = by_config[cfg]
        b["total"] += 1
        if r.get("compilation_ok"):
            b["compiled"] += 1
        if r.get("pass_at_1"):
            b["pass_at_1"] += 1
        b["latency_sum"] += r.get("latency_ms") or 0
        b["tokens_sum"] += r.get("total_tokens") or 0
        b["cost_sum"] += float(r.get("cost_usd") or 0)
        b["validation_sum"] += r.get("validation_attempts") or 0

        extra = r.get("extra_metrics") or {}
        if extra.get("guide_hit"):
            b["guide_hits"] += 1
        if extra.get("retrieval_top1_score") is not None:
            b["top1_sum"] += float(extra["retrieval_top1_score"])
            b["top1_count"] += 1

        va = r.get("validation_attempts") or 0
        if va > 1 and r.get("compilation_ok"):
            b["failed_first_recovered"] += 1
        if va > 1 or (va == 1 and not r.get("compilation_ok")):
            if va >= 1 and not r.get("pass_at_1"):
                b["failed_first_total"] += 1

    summary: dict[str, dict] = {}
    for cfg, b in by_config.items():
        total = b["total"]
        summary[cfg] = {
            "total_tasks": total,
            "accuracy_pct": round(b["compiled"] / total * 100, 1) if total else 0,
            "pass_at_1_pct": round(b["pass_at_1"] / total * 100, 1) if total else 0,
            "avg_latency_ms": round(b["latency_sum"] / total) if total else 0,
            "avg_tokens_per_task": round(b["tokens_sum"] / total) if total else 0,
            "total_cost_usd": round(b["cost_sum"], 4),
            "avg_validation_attempts": round(b["validation_sum"] / total, 2) if total else 0,
            "recovery_rate_pct": round(
                b["failed_first_recovered"] / b["failed_first_total"] * 100, 1
            ) if b["failed_first_total"] else 0.0,
            "guide_hit_rate_pct": round(b["guide_hits"] / total * 100, 1) if total else 0,
            "avg_retrieval_top1_score": round(
                b["top1_sum"] / b["top1_count"], 4
            ) if b["top1_count"] else 0.0,
        }

    if "vanilla_llm" in summary and "rag_skill_router" in summary:
        summary["rag_lift_pct"] = round(
            summary["rag_skill_router"]["accuracy_pct"]
            - summary["vanilla_llm"]["accuracy_pct"],
            1,
        )
    return summary


async def export_st_coding_run(
    run_id: str,
    db,
    *,
    guide_path: str | None = None,
    max_validation_attempts: int | None = None,
) -> dict:
    """Собрать полный JSON прогона из benchmark_results (после resume-пакетов)."""
    rows = await db.execute(
        text("""
            SELECT DISTINCT ON (task_id)
                   run_id::text AS run_id,
                   task_id,
                   difficulty,
                   config,
                   generated_code,
                   compilation_ok,
                   formal_verification_ok,
                   execution_correct,
                   matiec_errors,
                   total_tokens,
                   cost_usd,
                   latency_ms,
                   session_id::text AS session_id,
                   validation_attempts,
                   generate_attempts,
                   pass_at_1,
                   benchmark_suite,
                   extra_metrics
            FROM benchmark_results
            WHERE run_id = CAST(:rid AS uuid)
              AND benchmark_suite = 'st_coding'
            ORDER BY task_id, created_at DESC
        """),
        {"rid": run_id},
    )
    db_rows = rows.mappings().all()
    if not db_rows:
        return {"error": f"No st_coding results for run_id {run_id!r}"}

    results: list[dict] = []
    for r in db_rows:
        extra = r["extra_metrics"]
        if isinstance(extra, str):
            extra = json.loads(extra)
        merr = r["matiec_errors"]
        if isinstance(merr, str):
            merr = json.loads(merr)
        results.append({
            "run_id": r["run_id"],
            "task_id": r["task_id"],
            "difficulty": r["difficulty"],
            "config": r["config"],
            "generated_code": r["generated_code"],
            "compilation_ok": r["compilation_ok"],
            "formal_verification_ok": r["formal_verification_ok"],
            "execution_correct": r["execution_correct"],
            "matiec_errors": merr or [],
            "total_tokens": int(r["total_tokens"] or 0),
            "cost_usd": float(r["cost_usd"] or 0),
            "latency_ms": int(r["latency_ms"] or 0),
            "session_id": r["session_id"],
            "validation_attempts": r["validation_attempts"],
            "generate_attempts": r["generate_attempts"],
            "pass_at_1": r["pass_at_1"],
            "benchmark_suite": r["benchmark_suite"],
            "extra_metrics": extra or {},
        })

    config = results[0]["config"]
    session_ids = {r["session_id"] for r in results if r["session_id"]}
    max_va = max_validation_attempts or settings.benchmark_max_validation_attempts
    guide = resolve_guide_path(guide_path)

    return {
        "run_id": run_id,
        "suite": "st_coding",
        "config": config,
        "configs": [config],
        "session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
        "resumed": True,
        "skipped_task_ids": [],
        "task_ids_run": [r["task_id"] for r in results],
        "n_tasks": len(results),
        "max_validation_attempts": max_va,
        "guide": str(guide),
        "summary": _compute_summary(results),
        "results": results,
    }


async def run_st_coding_benchmark(
    n_tasks: int = 10,
    config: str | None = None,
    configs: list[str] | None = None,
    guide_path: str | None = None,
    max_validation_attempts: int | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    start_task_id: str | None = None,
    resume: bool = False,
    db=None,
) -> dict:
    try:
        resolved_configs = resolve_configs(config, configs)
    except ValueError as e:
        return {"error": str(e), "valid": sorted(VALID_CONFIGS)}

    config_name = resolved_configs[0]
    max_va = max_validation_attempts or settings.benchmark_max_validation_attempts
    all_tasks = load_tasks(n_tasks)
    if not all_tasks:
        return {"error": f"No tasks in {BENCH_FILE}"}

    resume_ctx = None
    if resume or session_id:
        resume_ctx = await get_resume_context(db, config_name, session_id=session_id)

    skip_ids: set[str] | None = None
    if resume:
        if not resume_ctx and not session_id:
            return {
                "error": "Nothing to resume: no prior run with session_id in DB",
                "config": config_name,
            }
        if resume_ctx and not start_task_id:
            skip_ids = resume_ctx["completed_task_ids"]
        if session_id is None and resume_ctx:
            session_id = resume_ctx["session_id"]
        if run_id is None and resume_ctx:
            run_id = resume_ctx["run_id"]

    try:
        tasks = filter_tasks(all_tasks, start_task_id=start_task_id, skip_task_ids=skip_ids)
    except ValueError as e:
        return {"error": str(e)}

    if not tasks:
        return {
            "error": "No tasks left to run (all completed or empty filter)",
            "session_id": session_id,
            "completed": sorted(skip_ids or []),
        }

    guide = resolve_guide_path(guide_path)
    guide_name = guide.name
    run_id = run_id or str(uuid.uuid4())
    all_results: list[dict] = []

    for cfg in resolved_configs:
        if cfg in {"rag_only", "rag_skills", "rag_skill_router"}:
            await _index_guide(db, guide, session_id=None, scope="global")
            for task in tasks:
                skills_list: list[str] | None = None
                route_skills_flag = False

                if cfg == "rag_skills":
                    from app.agents.react_agent import _skill_registry
                    if _skill_registry:
                        skills_list = _skill_registry.get_available_slugs()
                elif cfg == "rag_skill_router":
                    route_skills_flag = True

                sid = await ensure_session(db, str(uuid.uuid4()), title="ST coding benchmark")
                row = await _run_single(
                    run_id, task, cfg,
                    session_id=sid,
                    max_validation_attempts=max_va,
                    guide_name=guide_name,
                    db=db,
                    persist_chat=False,
                    skills=skills_list,
                    route_skills=route_skills_flag,
                )
                all_results.append(row)
        else:
            # vanilla_llm — no guide, no RAG, direct LLM
            for task in tasks:
                sid = await ensure_session(db, str(uuid.uuid4()), title="ST coding benchmark")
                row = await _run_single(
                    run_id, task, cfg,
                    session_id=sid,
                    max_validation_attempts=max_va,
                    guide_name=guide_name,
                    db=db,
                    persist_chat=False,
                )
                all_results.append(row)

    summary = _compute_summary(all_results)
    resumed_from = sorted(skip_ids) if skip_ids else []
    return {
        "run_id": run_id,
        "suite": "st_coding",
        "config": resolved_configs[0] if len(resolved_configs) == 1 else None,
        "configs": resolved_configs,
        "session_id": None,
        "resumed": bool(resume or start_task_id or session_id),
        "skipped_task_ids": resumed_from,
        "task_ids_run": [t["task_id"] for t in tasks],
        "n_tasks": len(tasks),
        "max_validation_attempts": max_va,
        "guide": str(guide),
        "summary": summary,
        "results": all_results,
    }
