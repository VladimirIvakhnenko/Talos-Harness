"""
app/monitoring/token_tracker.py — LangChain callback для логирования токенов.
"""
from __future__ import annotations
import asyncio
import time
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from app.config import get_settings

settings = get_settings()


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self, session_id=None, agent_name="unknown",
                 tool_name=None, task_id=None, react_step=None):
        super().__init__()
        self.session_id = session_id
        self.agent_name = agent_name
        self.tool_name  = tool_name
        self.task_id    = task_id
        self.react_step = react_step
        self._t0: float = 0.0

    def on_llm_start(self, serialized, prompts, **kw):
        self._t0 = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kw):
        latency = int((time.perf_counter() - self._t0) * 1000)
        usage   = {}
        if response.llm_output:
            usage = (response.llm_output.get("token_usage")
                     or response.llm_output.get("usage", {}))

        prompt_t = usage.get("prompt_tokens", 0)
        comp_t   = usage.get("completion_tokens", 0)
        total_t  = usage.get("total_tokens", prompt_t + comp_t)
        model_id = ""
        if response.llm_output:
            model_id = (response.llm_output.get("model_name")
                        or response.llm_output.get("model", ""))
        cost = settings.cost_usd(model_id, prompt_t, comp_t)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._save(model_id, prompt_t, comp_t, total_t, cost, latency))
            else:
                loop.run_until_complete(self._save(model_id, prompt_t, comp_t, total_t, cost, latency))
        except Exception:
            pass

    async def _save(self, model_id, prompt_t, comp_t, total_t, cost, latency):
        from sqlalchemy import text
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                INSERT INTO token_usage
                    (session_id,agent_name,tool_name,model_id,
                     prompt_tokens,completion_tokens,total_tokens,
                     cost_usd,latency_ms,react_step,task_id)
                VALUES
                    (:sid,:agent,:tool,:model,
                     :pt,:ct,:tt,:cost,:lat,:step,:task)
            """), dict(
                sid=self.session_id, agent=self.agent_name, tool=self.tool_name,
                model=model_id, pt=prompt_t, ct=comp_t, tt=total_t,
                cost=cost, lat=latency, step=self.react_step, task=self.task_id,
            ))
            await db.commit()