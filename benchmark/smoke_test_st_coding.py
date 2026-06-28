"""Smoke-test: load tasks, summary math, optional live API call."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.st_coding_runner import _compute_summary, load_tasks


def test_load_and_summary():
    tasks = load_tasks(2)
    assert len(tasks) == 2, tasks
    assert tasks[0]["task_id"] == "IA01"

    mock = [
        {
            "config": "baseline",
            "compilation_ok": True,
            "pass_at_1": True,
            "latency_ms": 1000,
            "total_tokens": 0,
            "cost_usd": 0,
            "validation_attempts": 0,
            "extra_metrics": {},
        },
        {
            "config": "agent_single_session",
            "compilation_ok": False,
            "pass_at_1": False,
            "latency_ms": 5000,
            "total_tokens": 1200,
            "cost_usd": 0.01,
            "validation_attempts": 2,
            "extra_metrics": {"guide_hit": True, "retrieval_top1_score": 0.8},
        },
    ]
    summary = _compute_summary(mock)
    assert "baseline" in summary
    assert summary["baseline"]["accuracy_pct"] == 100.0
    assert summary["agent_single_session"]["guide_hit_rate_pct"] == 100.0
    print("OK: load_tasks + _compute_summary")


async def test_live_api():
    import urllib.request

    payload = json.dumps({
        "config": "baseline",
        "n_tasks": 1,
        "max_validation_attempts": 2,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/benchmark/st_coding/run",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        print(f"SKIP live API (service unavailable): {e}")
        return

    assert body.get("suite") == "st_coding"
    assert "summary" in body
    assert "results" in body
    r0 = body["results"][0]
    for key in ("task_id", "compilation_ok", "latency_ms", "benchmark_suite"):
        assert key in r0, f"missing {key} in {r0.keys()}"
    print("OK: live API", json.dumps(body["summary"], ensure_ascii=False))


if __name__ == "__main__":
    test_load_and_summary()
    asyncio.run(test_live_api())
