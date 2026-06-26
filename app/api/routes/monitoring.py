from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from app.database import get_db

router = APIRouter(tags=["Monitoring"])


@router.get("/monitoring/tokens", summary="Суммарные токены по сессиям и агентам")
async def monitoring_tokens(session_id: Optional[str] = Query(None)):
    async with get_db() as db:
        if session_id:
            rows = await db.execute(
                text("SELECT * FROM v_token_summary WHERE session_id=:sid"),
                {"sid": session_id},
            )
        else:
            rows = await db.execute(
                text("SELECT * FROM v_token_summary ORDER BY last_call_at DESC LIMIT 100")
            )
        return {"tokens": [dict(r._mapping) for r in rows.fetchall()]}


@router.get("/monitoring/cost", summary="Стоимость вызовов в USD с разбивкой по агентам")
async def monitoring_cost(
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
):
    async with get_db() as db:
        rows = await db.execute(
            text("""
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
        """),
            {"from_d": from_date, "to_d": to_date},
        )
        return {"cost_breakdown": [dict(r._mapping) for r in rows.fetchall()]}


@router.get("/monitoring/cost/per_task", summary="Средняя стоимость задачи Agents4PLC по сложности и конфигурации")
async def cost_per_task():
    async with get_db() as db:
        rows = await db.execute(
            text("""
            SELECT b.config, b.difficulty,
                   COUNT(*) AS tasks,
                   ROUND(AVG(t.cost_usd)::NUMERIC*1000,4) AS avg_cost_per_task_milli_usd
            FROM benchmark_results b
            LEFT JOIN token_usage t ON t.task_id = b.task_id
            GROUP BY b.config, b.difficulty
            ORDER BY b.config, b.difficulty
        """)
        )
        return {"per_task": [dict(r._mapping) for r in rows.fetchall()]}


@router.get(
    "/monitoring/dashboard",
    response_class=HTMLResponse,
    summary="Plotly HTML-дашборд: токены / стоимость / latency",
)
async def monitoring_dashboard():
    async with get_db() as db:
        rows = await db.execute(
            text("""
            SELECT agent_name, model_id,
                   SUM(total_tokens) AS tokens,
                   ROUND(SUM(cost_usd)::NUMERIC,6) AS cost,
                   AVG(latency_ms)::INT AS avg_lat
            FROM token_usage GROUP BY agent_name, model_id ORDER BY tokens DESC
        """)
        )
        data = [dict(r._mapping) for r in rows.fetchall()]

    agents = [d["agent_name"] or "unknown" for d in data]
    tokens = [int(d["tokens"] or 0) for d in data]
    costs = [float(d["cost"] or 0) for d in data]
    latency = [int(d["avg_lat"] or 0) for d in data]

    html = f"""<!DOCTYPE html><html>
<head><title>Talos Harness — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:Arial,sans-serif;background:#0f1117;color:#eee;padding:20px;margin:0}}
h1{{color:#4BACC6;}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}}
.card{{background:#1e2130;border-radius:12px;padding:16px}}</style></head>
<body>
<h1>Talos Harness — Token & Cost Dashboard</h1>
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
