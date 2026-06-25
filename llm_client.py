"""
app/agents/llm_client.py — LLM через OpenRouter (primary, единственный провайдер).
"""
from __future__ import annotations
from langchain_openai import ChatOpenAI
from app.config import get_settings
from app.monitoring.token_tracker import TokenUsageCallback

settings = get_settings()

_MODEL_MAP = {
    "planner":   settings.planner_model,
    "engineer":  settings.engineer_model,
    "retriever": settings.retriever_model,
}


def get_llm(
    role: str = "engineer",
    session_id: str | None = None,
    task_id: str | None = None,
    tool_name: str | None = None,
    react_step: str | None = None,
    temperature: float | None = None,
):
    """Фабрика LLM через OpenRouter с прикреплённым TokenUsageCallback."""
    cb = TokenUsageCallback(
        session_id=session_id,
        agent_name=role,
        tool_name=tool_name,
        task_id=task_id,
        react_step=react_step,
    )
    return ChatOpenAI(
        model=_MODEL_MAP.get(role, settings.engineer_model),
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature if temperature is not None else settings.agent_temperature,
        max_tokens=4096,
        callbacks=[cb],
        default_headers={
            "HTTP-Referer": "https://github.com/talos-harness",
            "X-Title":      "Talos Harness PLC Agent",
        },
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Батчевая векторизация через text-embedding-3-small (OpenRouter)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    resp = await client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]


async def embed_single(text: str) -> list[float]:
    return (await embed_texts([text]))[0]