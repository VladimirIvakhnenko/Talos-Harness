from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://plc:plc_secret@localhost:5432/plc_agent"

    # LLM provider: openrouter | lmstudio
    llm_backend: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"

    # Models
    planner_model: str   = "qwen/qwen3.6-27b"
    engineer_model: str  = "qwen/qwen3.6-27b"
    retriever_model: str = "qwen/qwen3.6-27b"
    embedding_backend: str = "local"  # local (llama.cpp GGUF) | openrouter
    embedding_model: str = "embeddinggemma-300m"
    embedding_dimensions: int = 768
    embedding_max_chars: int = 1500
    embedding_gguf_file: str = "embeddinggemma-300M-Q8_0.gguf"
    llama_embedding_url: str = "http://localhost:8080"

    # MatIEC
    matiec_url: str = "http://localhost:8001"

    # Agent (context_limit = input + output; max_tokens is output only)
    agent_max_iterations: int = 10
    expert_max_iterations: int = 5
    agent_context_limit: int = 131_072
    agent_input_reserve: int = 16_384
    agent_max_tokens: int = 120_000
    agent_temperature: float  = 0.1
    chunk_size: int    = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5

    # OCR
    ocr_backend: str = "openrouter"  # paddle | openrouter
    ocr_model: str = "qwen/qwen3-vl-8b-thinking"
    ocr_lang: str = "ru"
    ocr_version: str = "PP-OCRv5"
    ocr_use_server_models: bool = True
    ocr_min_score: float = 0.5

    # App
    app_env: str    = "development"
    log_level: str  = "INFO"
    upload_dir: str = "/app/uploads"

    # Pricing USD per 1M tokens (OpenRouter, June 2026)
    model_pricing: dict = {
        "nex-agi/nex-n2-pro":     {"input": 0.0,  "output": 0.0},
        "qwen/qwen3.6-27b":       {"input": 0.0,  "output": 0.0},
        "openai/gpt-oss-120b":    {"input": 0.0,  "output": 0.0},
        "qwen/qwen3.5-9b":        {"input": 0.1,  "output": 0.2},
        "qwen/qwen3.5-4b":        {"input": 0.05, "output": 0.1},
        "poolside/laguna-m1":     {"input": 0.0,  "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    def cost_usd(self, model_id: str, prompt_t: int, comp_t: int) -> float:
        p = self.model_pricing.get(model_id, {"input": 0.5, "output": 1.5})
        return prompt_t * p["input"] / 1_000_000 + comp_t * p["output"] / 1_000_000

    def llm_api_base(self) -> str:
        if self.llm_backend.lower() == "lmstudio":
            return self.lmstudio_base_url
        return self.openrouter_base_url

    def llm_api_key(self) -> str:
        if self.llm_backend.lower() == "lmstudio":
            return self.lmstudio_api_key
        return self.openrouter_api_key

    def completion_token_limit(self) -> int:
        """Бюджет на ответ: context_limit − резерв под промпт/инструменты."""
        budget = self.agent_context_limit - self.agent_input_reserve
        return min(self.agent_max_tokens, max(budget, 4096))


@lru_cache
def get_settings() -> Settings:
    return Settings()