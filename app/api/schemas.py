from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="Запрос пользователя на естественном языке")
    session_id: Optional[str] = Field(None, description="UUID сессии; если не задан — создаётся автоматически")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Напиши функциональный блок для управления насосом с защитой от сухого хода",
                "session_id": None,
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


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str]
