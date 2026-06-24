"""
app/config.py — Централизованная конфигурация через pydantic-settings.
Читает из .env / переменных окружения.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://plc:plc_secret@localhost:5432/plc_agent"

    # ── OpenRouter ──────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Модели по ролям ─────────────────────────────────────────────────────
    planner_model: str = "nex-agi/nex-n2-pro"
    engineer_model: str = "qwen/qwen3.5-9b"
    retriever_model: str = "qwen/qwen3.5-4b"
    embedding_model: str = "text-embedding-3-small"

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:9b"

    # ── Режим LLM ───────────────────────────────────────────────────────────
    llm_provider: str = "openrouter"   # "openrouter" | "ollama"

    # ── Параметры агента ────────────────────────────────────────────────────
    agent_max_iterations: int = 10
    agent_temperature: float = 0.1
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5

    # ── Приложение ──────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    upload_dir: str = "/app/uploads"

    # ── Стоимость моделей (USD за 1M токенов) ───────────────────────────────
    # Источник: openrouter.ai/models (актуально на июнь 2026)
    model_pricing: dict = {
        "nex-agi/nex-n2-pro":    {"input": 0.0,  "output": 0.0},   # бесплатная
        "qwen/qwen3.5-9b":       {"input": 0.1,  "output": 0.2},
        "qwen/qwen3.5-4b":       {"input": 0.05, "output": 0.1},
        "poolside/laguna-m1":    {"input": 0.0,  "output": 0.0},   # бесплатная
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    def cost_usd(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Рассчитать стоимость вызова LLM в USD."""
        pricing = self.model_pricing.get(model_id, {"input": 0.5, "output": 1.5})
        return (
            prompt_tokens     * pricing["input"]  / 1_000_000 +
            completion_tokens * pricing["output"] / 1_000_000
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()