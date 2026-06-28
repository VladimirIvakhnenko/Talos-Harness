"""ST coding benchmark — один прогон на одну config (10 задач).

Примеры:
  python benchmark/run_full_bench.py --config baseline
  python benchmark/run_full_bench.py --config agent_isolated
  python benchmark/run_full_bench.py --config agent_single_session

Продолжить прерванный single-session прогон в том же чате:
  python benchmark/run_full_bench.py --config agent_single_session --resume
  python benchmark/run_full_bench.py --config agent_single_session \\
      --session-id 1d154fc2-b429-4f16-8db9-6786778837df --start-task IA06

Три полных сравнения = три отдельных запуска (30 прогонов суммарно).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import get_db
from benchmark.st_coding_runner import ALL_CONFIGS, export_st_coding_run, run_st_coding_benchmark


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ST coding benchmark (single config run)")
    p.add_argument(
        "--config",
        required=True,
        choices=ALL_CONFIGS,
        help="Режим прогона: baseline | agent_isolated | agent_single_session",
    )
    p.add_argument("--n-tasks", type=int, default=10, help="Число задач из st_coding_bench.json")
    p.add_argument("--max-validation-attempts", type=int, default=2)
    p.add_argument("--guide-path", default=None, help="Путь к ST-гайду (MD)")
    p.add_argument(
        "--output",
        default=None,
        help="Файл результата (по умолчанию st_coding_run_<config>.json)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить последний прогон: тот же session_id/run_id, пропустить выполненные задачи",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="UUID сессии для agent_single_session (история чата сохраняется)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="UUID прогона в benchmark_results (по умолчанию берётся из --resume)",
    )
    p.add_argument(
        "--start-task",
        default=None,
        dest="start_task_id",
        help="Начать с task_id (например IA06); при --resume без --start-task уже выполненные пропускаются",
    )
    p.add_argument(
        "--export-run-id",
        default=None,
        help="Экспорт полного прогона из БД в JSON (без запуска задач)",
    )
    return p.parse_args()


async def _export_run(args: argparse.Namespace, out: Path) -> None:
    async with get_db() as db:
        result = await export_st_coding_run(
            args.export_run_id,
            db,
            guide_path=args.guide_path,
            max_validation_attempts=args.max_validation_attempts,
        )
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{datetime.now().isoformat()}] Exported run {args.export_run_id} to {out}", flush=True)
    if "error" in result:
        print(f"ERROR: {result['error']}", flush=True)
        sys.exit(1)
    print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False), flush=True)


async def main() -> None:
    args = _parse_args()
    out = Path(args.output) if args.output else Path(__file__).with_name(
        f"st_coding_run_{args.config}.json"
    )

    if args.export_run_id:
        await _export_run(args, out)
        return

    print(f"[{datetime.now().isoformat()}] ST coding benchmark — config={args.config}", flush=True)
    print(f"Tasks: {args.n_tasks}, max_validation_attempts: {args.max_validation_attempts}", flush=True)
    if args.resume:
        print("Resume: enabled (same chat session)", flush=True)
    if args.session_id:
        print(f"Session: {args.session_id}", flush=True)
    if args.start_task_id:
        print(f"Start task: {args.start_task_id}", flush=True)

    async with get_db() as db:
        result = await run_st_coding_benchmark(
            n_tasks=args.n_tasks,
            config=args.config,
            guide_path=args.guide_path,
            max_validation_attempts=args.max_validation_attempts,
            session_id=args.session_id,
            run_id=args.run_id,
            start_task_id=args.start_task_id,
            resume=args.resume,
            db=db,
        )

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{datetime.now().isoformat()}] Done. Saved to {out}", flush=True)

    if "error" in result:
        print(f"ERROR: {result['error']}", flush=True)
        sys.exit(1)

    print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
