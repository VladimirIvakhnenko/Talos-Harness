"""
app/monitoring/token_tracker.py — LangChain callback для логирования токенов.

Поддерживает: OpenAI / OpenRouter / vLLM (LM Studio).
Извлекает usage из llm_output, generation_info, response_metadata.
Если API не вернул usage — считает через tiktoken (офлайн-оценка).
"""
from __future__ import annotations
import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_tiktoken_enc: Any = None


def _get_encoder(model: str) -> Any:
    global _tiktoken_enc
    if _tiktoken_enc is not None:
        return _tiktoken_enc
    try:
        import tiktoken
        _tiktoken_enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = False
    return _tiktoken_enc


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = _get_encoder(model)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _extract_model_id(response: LLMResult) -> str:
    """Пробует все источники для извлечения model_id."""
    if response.llm_output:
        mid = response.llm_output.get("model_name") or response.llm_output.get("model") or ""
        if mid:
            return mid
    for gen_list in response.generations:
        for gen in gen_list:
            info = gen.generation_info or {}
            if isinstance(info, dict):
                mid = info.get("model") or info.get("model_name") or ""
                if mid:
                    return mid
            meta = getattr(gen, "message", None)
            if meta:
                meta = getattr(meta, "response_metadata", {})
            else:
                meta = getattr(gen, "response_metadata", {})
            if isinstance(meta, dict):
                mid = meta.get("model") or meta.get("model_name") or ""
                if mid:
                    return mid
    return ""


def _extract_token_usage(response: LLMResult) -> tuple[int, int, int]:
    """Извлекает prompt_tokens, completion_tokens, total_tokens
    из LLMResult / ChatResult, пробуя все возможные источники.

    Возвращает (prompt_t, comp_t, total_t) — все 0 если не найдены.
    """
    prompt_t = 0
    comp_t = 0
    total_t = 0

    # 1. llm_output (OpenAI / OpenRouter)
    if response.llm_output:
        token_source = (
            response.llm_output.get("token_usage")
            or response.llm_output.get("usage", {})
        )
        if isinstance(token_source, dict):
            prompt_t = token_source.get("prompt_tokens", 0) or 0
            comp_t = token_source.get("completion_tokens", 0) or 0
            total_t = token_source.get("total_tokens", prompt_t + comp_t) or 0
        if prompt_t or comp_t:
            return prompt_t, comp_t, total_t

    # 2. generation_info (vLLM, некоторые прокси)
    for gen_list in response.generations:
        for gen in gen_list:
            info = gen.generation_info or {}
            if not isinstance(info, dict):
                continue
            pt = info.get("prompt_tokens", info.get("input_tokens", 0)) or 0
            ct = info.get("completion_tokens", info.get("output_tokens", 0)) or 0
            prompt_t += pt
            comp_t += ct
            total_t += info.get("total_tokens", pt + ct) or 0

    if prompt_t or comp_t:
        return prompt_t, comp_t, total_t

    # 3. response_metadata (новые версии LangChain)
    for gen_list in response.generations:
        for gen in gen_list:
            meta = getattr(gen, "message", None)
            if meta:
                meta = getattr(meta, "response_metadata", {})
            else:
                meta = getattr(gen, "response_metadata", {})
            if not isinstance(meta, dict):
                continue
            pt = meta.get("prompt_tokens", meta.get("input_tokens", 0)) or 0
            ct = meta.get("completion_tokens", meta.get("output_tokens", 0)) or 0
            prompt_t += pt
            comp_t += ct
            total_t += meta.get("total_tokens", pt + ct) or 0

    return prompt_t, comp_t, total_t


class TokenUsageCollector:
    """Потокобезопасный on-the-fly коллектор токенов (без БД).
    Аккумулирует расход в памяти, отдаёт сбросом по требованию.
    """
    _acc: dict[str, int]

    def __init__(self):
        self.reset()

    def add(self, model_id: str, prompt_t: int, comp_t: int, total_t: int, cost: float):
        if not prompt_t and not comp_t:
            return
        self._acc["calls"] += 1
        self._acc["prompt_tokens"] += prompt_t
        self._acc["completion_tokens"] += comp_t
        self._acc["total_tokens"] += total_t
        self._acc["cost_usd"] += cost

    def snapshot(self) -> dict:
        """Возвращает и сбрасывает накопленное."""
        s = dict(self._acc)
        self.reset()
        return s

    def reset(self):
        self._acc = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}


# ── Module-level cycle collector для per-ReAct отображения ──────────
_cycle_collector = TokenUsageCollector()


def drain_cycle_tokens() -> dict:
    """Сброс накопленного за один ReAct-цикл (для UI / стрима)."""
    return _cycle_collector.snapshot()


# ── Pending rows buffer (синхронное накопление для бенчмарков) ──────
_pending_rows: list[dict] = []


async def flush_pending_tokens(db, session_id: str | None = None, task_id: str | None = None) -> None:
    """Сброс накопленных строк token_usage в БД.

    Если session_id / task_id переданы — сбрасывает только строки для них.
    Иначе — сбрасывает всё.
    """
    global _pending_rows
    if not _pending_rows:
        return

    rows = list(_pending_rows)
    logger.info(
        "flush_pending_tokens: %d pending rows, filter sid=%s tid=%s",
        len(rows), session_id, task_id,
    )

    # Если фильтр задан — отфильтровать только подходящие строки, остальные оставить
    if session_id is not None or task_id is not None:
        matched = []
        remaining = []
        for r in rows:
            if (session_id is None or r.get("session_id") == str(session_id)) and \
               (task_id is None or r.get("task_id") == task_id):
                matched.append(r)
            else:
                remaining.append(r)
        if not matched:
            return
        rows = matched
        _pending_rows = remaining
    else:
        _pending_rows = []

    from sqlalchemy import text
    for row in rows:
        try:
            await db.execute(text("""
                INSERT INTO token_usage
                    (session_id, agent_name, tool_name, model_id,
                     prompt_tokens, completion_tokens, total_tokens,
                     cost_usd, latency_ms, react_step, task_id)
                VALUES
                    (:sid, :agent, :tool, :model,
                     :pt, :ct, :tt, :cost, :lat, :step, :task)
            """), dict(
                sid=row.get("session_id"),
                agent=row.get("agent_name"),
                tool=row.get("tool_name"),
                model=row.get("model_id"),
                pt=row.get("prompt_tokens", 0),
                ct=row.get("completion_tokens", 0),
                tt=row.get("total_tokens", 0),
                cost=row.get("cost_usd", 0.0),
                lat=row.get("latency_ms"),
                step=row.get("react_step"),
                task=row.get("task_id"),
            ))
        except Exception:
            logger.warning("Failed to flush token row", exc_info=True)
    try:
        await db.commit()
    except Exception:
        logger.warning("Failed to commit token flush", exc_info=True)


class TokenUsageCallback(BaseCallbackHandler):
    """LangChain callback: логирует токены, latency и стоимость в token_usage.

    Если API (vLLM / OpenRouter) не вернул usage — считает через tiktoken.
    """

    def __init__(self, session_id=None, agent_name="unknown",
                 tool_name=None, task_id=None, react_step=None):
        super().__init__()
        self.session_id = session_id
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.task_id = task_id
        self.react_step = react_step
        self._t0: float = 0.0
        self._prompts: list[str] = []

    def on_llm_start(self, serialized: dict, prompts: list[str], **kw):
        self._t0 = time.perf_counter()
        self._prompts = prompts

    def on_llm_end(self, response: LLMResult, **kw):
        latency = int((time.perf_counter() - self._t0) * 1000)
        model_id = _extract_model_id(response)
        prompt_t, comp_t, total_t = _extract_token_usage(response)

        logger.info(
            "TokenUsageCallback: model=%s pt=%s ct=%s tt=%s sid=%s tid=%s",
            model_id, prompt_t, comp_t, total_t,
            self.session_id, self.task_id,
        )

        # Fallback: offline-подсчёт через tiktoken
        if not prompt_t and not comp_t:
            full_prompt = "\n".join(self._prompts) if self._prompts else ""
            completion = ""
            for gen_list in response.generations:
                for gen in gen_list:
                    completion += (gen.text or "") + "\n"
            prompt_t = _count_tokens(full_prompt, model_id)
            comp_t = _count_tokens(completion.strip(), model_id)
            total_t = prompt_t + comp_t
            logger.debug(
                "Token usage fallback (tiktoken): pt=%s ct=%s model=%s",
                prompt_t, comp_t, model_id,
            )

        cost = settings.cost_usd(model_id, prompt_t, comp_t)

        # Feed module-level cycle collector (для streaming UI)
        _cycle_collector.add(model_id, prompt_t, comp_t, total_t, cost)

        # Append to pending buffer (синхронно, без fire-and-forget)
        _pending_rows.append({
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "model_id": model_id,
            "prompt_tokens": prompt_t,
            "completion_tokens": comp_t,
            "total_tokens": total_t,
            "cost_usd": cost,
            "latency_ms": latency,
            "react_step": self.react_step,
            "task_id": self.task_id,
        })
