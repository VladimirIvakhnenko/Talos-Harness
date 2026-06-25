from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://plc:plc_secret@localhost:5432/plc_agent"

    # OpenRouter — PRIMARY provider
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Models
    planner_model: str   = "nex-agi/nex-n2-pro"
    engineer_model: str  = "qwen/qwen3.5-9b"
    retriever_model: str = "qwen/qwen3.5-4b"
    embedding_model: str = "text-embedding-3-small"

    # MatIEC
    matiec_url: str = "http://localhost:8001"

    # Agent
    agent_max_iterations: int = 10
    agent_temperature: float  = 0.1
    chunk_size: int    = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5

    # App
    app_env: str    = "development"
    log_level: str  = "INFO"
    upload_dir: str = "/app/uploads"

    # Pricing USD per 1M tokens (OpenRouter, June 2026)
    model_pricing: dict = {
        "nex-agi/nex-n2-pro":     {"input": 0.0,  "output": 0.0},
        "qwen/qwen3.5-9b":        {"input": 0.1,  "output": 0.2},
        "qwen/qwen3.5-4b":        {"input": 0.05, "output": 0.1},
        "poolside/laguna-m1":     {"input": 0.0,  "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    def cost_usd(self, model_id: str, prompt_t: int, comp_t: int) -> float:
        p = self.model_pricing.get(model_id, {"input": 0.5, "output": 1.5})
        return prompt_t * p["input"] / 1_000_000 + comp_t * p["output"] / 1_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()