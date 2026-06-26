"""
app/tools/matiec_client.py — HTTP-клиент к MatIEC-сервису.

Агент вызывает validate_st() и compile_st() напрямую.
"""
from __future__ import annotations
import httpx
from app.config import get_settings

settings = get_settings()


class MatIECResult:
    def __init__(self, data: dict):
        self.ok               = data.get("ok", False)
        self.compilation_rate = data.get("compilation_rate", 0.0)
        self.errors           = data.get("errors", [])
        self.warnings         = data.get("warnings", [])
        self.stdout           = data.get("stdout", "")
        self.stderr           = data.get("stderr", "")

    def __repr__(self):
        status = "OK" if self.ok else f"FAIL ({len(self.errors)} errors)"
        return f"MatIECResult({status})"


async def validate_st(code: str, task_id: str = "unnamed") -> MatIECResult:
    """Синтаксическая проверка ST-кода (быстро)."""
    return await _call("/validate", code, task_id)


async def compile_st(code: str, task_id: str = "unnamed") -> MatIECResult:
    """Полная компиляция ST-кода (медленнее)."""
    return await _call("/compile", code, task_id)


async def _call(endpoint: str, code: str, task_id: str) -> MatIECResult:
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            resp = await client.post(
                settings.matiec_url + endpoint,
                json={"code": code, "task_id": task_id},
            )
            resp.raise_for_status()
            return MatIECResult(resp.json())
    except httpx.ConnectError:
        return MatIECResult({
            "ok": False, "compilation_rate": 0.0,
            "errors": [f"MatIEC service unavailable at {settings.matiec_url}"],
            "warnings": [], "stdout": "", "stderr": "",
        })
    except Exception as e:
        return MatIECResult({
            "ok": False, "compilation_rate": 0.0,
            "errors": [str(e)], "warnings": [], "stdout": "", "stderr": "",
        })


async def matiec_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(settings.matiec_url + "/health")
            return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}