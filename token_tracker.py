"""
app/monitoring/token_tracker.py — LangChain callback для логирования токенов.

Перехватывает каждый вызов LLM и записывает в token_usage.
Поддерживает OpenRouter и Ollama.
"""
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import get_settings
from app.database import AsyncSessionLocal

settings = get_settings()


class TokenUsageCallback(BaseCallbackHandler):
    """
    LangChain callback: перехватывает on_llm_start / on_llm_end,
    вычисляет стоимость и записывает строку в token_usage.
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_name: str = "unknown",
        tool_name: str | None = None,
        task_id: str | None = None,
        react_step: str | None = None,
    ):
        super().__init__()
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.task_id = task_id
        self.react_step = react_step
        self._start_time: float = 0.0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency_ms = int((time.perf_counter() - self._start_time) * 1000)

        # Извлекаем токены из llm_output (OpenRouter / OpenAI формат)
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            # Ollama возвращает чуть иначе
            if not usage:
                usage = response.llm_output.get("usage", {})

        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Определяем модель
        model_id = ""
        if response.llm_output:
            model_id = response.llm_output.get("model_name", "")
            if not model_id:
                model_id = response.llm_output.get("model", "")

        cost_usd = settings.cost_usd(model_id, prompt_tokens, completion_tokens)

        # Асинхронная запись через синхронный вызов (callback синхронный)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Уже в event loop — создаём задачу
                loop.create_task(
                    self._save_to_db(
                        model_id, prompt_tokens, completion_tokens,
                        total_tokens, cost_usd, latency_ms,
                    )
                )
            else:
                loop.run_until_complete(
                    self._save_to_db(
                        model_id, prompt_tokens, completion_tokens,
                        total_tokens, cost_usd, latency_ms,
                    )
                )
        except RuntimeError:
            # Fallback: просто логируем
            import logging
            logging.getLogger(__name__).warning(
                "TokenUsageCallback: cannot save to DB. "
                f"tokens={total_tokens} cost=${cost_usd:.6f}"
            )

    async def _save_to_db(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        latency_ms: int,
    ) -> None:
        """Записывает строку в token_usage."""
        from sqlalchemy import text

        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    INSERT INTO token_usage
                        (session_id, agent_name, tool_name, model_id,
                         prompt_tokens, completion_tokens, total_tokens,
                         cost_usd, latency_ms, react_step, task_id)
                    VALUES
                        (:session_id, :agent_name, :tool_name, :model_id,
                         :prompt_tokens, :completion_tokens, :total_tokens,
                         :cost_usd, :latency_ms, :react_step, :task_id)
                """),
                {
                    "session_id":         self.session_id,
                    "agent_name":         self.agent_name,
                    "tool_name":          self.tool_name,
                    "model_id":           model_id,
                    "prompt_tokens":      prompt_tokens,
                    "completion_tokens":  completion_tokens,
                    "total_tokens":       total_tokens,
                    "cost_usd":           cost_usd,
                    "latency_ms":         latency_ms,
                    "react_step":         self.react_step,
                    "task_id":            self.task_id,
                },
            )
            await db.commit()