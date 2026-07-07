from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="Запрос пользователя на естественном языке")
    session_id: Optional[str] = Field(None, description="UUID сессии; если не задан — создаётся автоматически")
    skills: Optional[list[str]] = Field(None, description="Список скиллов для активации в этом запросе")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Напиши функциональный блок для управления насосом с защитой от сухого хода",
                "session_id": None,
                "skills": None,
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    response: str
    final_code: Optional[str] = None
    matiec_ok: Optional[bool] = None
    steps: int = 0


class GenerateModuleRequest(BaseModel):
    controller: str = Field("elbrus", description="elbrus | baikal | codesys")
    signals_path: Optional[str] = Field(None, description="Путь к CSV с сигналами (после /upload_signals)")
    module_name: str = Field("GeneratedModule", description="Имя PROGRAM или FUNCTION_BLOCK")
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "controller": "elbrus",
                "signals_path": "/app/uploads/signals_abc.csv",
                "module_name": "PumpController",
            }
        }


class GenerateModuleResponse(BaseModel):
    session_id: str
    controller: str
    module_name: str
    code: str
    matiec_ok: Optional[bool] = None
    matiec_errors: list[str] = []
    download_url: str


class ValidateRequest(BaseModel):
    code: str = Field(..., description="ST-код для верификации")
    task_id: str = Field("manual", description="Идентификатор задачи")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "PROGRAM Test\nVAR x: BOOL; END_VAR\nx := TRUE;\nEND_PROGRAM",
                "task_id": "test_001",
            }
        }


class ValidateResponse(BaseModel):
    ok: bool
    compilation_rate: float
    errors: list[str]
    warnings: list[str]


class BenchmarkRunRequest(BaseModel):
    subset: str = Field("medium", description="easy | medium | hard")
    n_tasks: int = Field(10, ge=1, le=96, description="Количество задач для оценки")
    configs: list[str] = Field(["baseline", "full_agent"], description="Конфигурации для сравнения")

    class Config:
        json_schema_extra = {
            "example": {"subset": "medium", "n_tasks": 10, "configs": ["baseline", "full_agent"]}
        }


class StCodingBenchRunRequest(BaseModel):
    config: str = Field(
        "vanilla_llm",
        description="Один режим прогона: vanilla_llm | rag_only | rag_skills | rag_skill_router",
    )
    n_tasks: int = Field(10, ge=1, le=50, description="Количество задач из st_coding_bench.json")
    configs: Optional[list[str]] = Field(
        None,
        description="Устарело: используйте config. Если задано — переопределяет config.",
    )
    guide_path: Optional[str] = Field(
        None,
        description="Путь к ST-гайду (MD). По умолчанию benchmark/assets/IEC-61131-3-ST-GUIDE.md",
    )
    max_validation_attempts: int = Field(2, ge=1, le=10, description="Лимит попыток validate_st_syntax")
    route_skills: bool = Field(
        False,
        description="Автоматический выбор скиллов через cosine similarity (для rag_skill_router)",
    )
    run_id: Optional[str] = Field(
        None,
        description="UUID прогона в benchmark_results (для склейки метрик)",
    )
    start_task_id: Optional[str] = Field(
        None,
        description="Начать с task_id (например IA06)",
    )
    resume: bool = Field(
        False,
        description="Продолжить последний прогон: тот же session/run, пропустить выполненные задачи",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "config": "agent_single_session",
                "n_tasks": 10,
                "max_validation_attempts": 2,
                "resume": True,
            }
        }


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str]


# ---- Skills ----

class SkillInfo(BaseModel):
    slug: str
    name: str
    version: str
    description: str
    has_tools: bool
    has_nodes: bool
    active: bool
    legacy: bool = False
    prompt_preview: str = ""


class SkillDetail(SkillInfo):
    license: str = ""
    depends_on: list[str] = []
    tools: list[str] = []
    prompt_body: str = ""


class SkillActivateResponse(BaseModel):
    slug: str
    active: bool
    message: str


class SkillUploadResponse(BaseModel):
    slug: str
    name: str
    description: str
    version: str
    message: str
