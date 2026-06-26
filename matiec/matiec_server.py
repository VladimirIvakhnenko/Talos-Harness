"""
matiec_server.py — HTTP-обёртка над MatIEC (iec2c / iec2iec).

Эндпоинты:
  POST /compile   — компиляция ST-кода, возвращает {ok, errors, warnings}
  POST /validate  — синтаксическая проверка без генерации C-кода
  GET  /health    — статус сервиса
"""
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MatIEC Server", version="1.0.0")

WORK_DIR = Path(os.getenv("MATIEC_WORK_DIR", "/tmp/matiec_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

IEC2C   = "/usr/local/bin/iec2c"
IEC2IEC = "/usr/local/bin/iec2iec"


class CompileRequest(BaseModel):
    code: str
    task_id: str = "unnamed"


class CompileResult(BaseModel):
    ok: bool
    compilation_rate: float        # 1.0 или 0.0
    errors: list[str]
    warnings: list[str]
    stdout: str
    stderr: str


MATIEC_LIB = Path("/usr/local/lib/matiec")


@app.get("/health")
def health():
    iec2c_ok   = Path(IEC2C).exists()
    iec2iec_ok = Path(IEC2IEC).exists()
    lib_ok     = MATIEC_LIB.is_dir() and any(MATIEC_LIB.iterdir())
    ok         = iec2c_ok and iec2iec_ok and lib_ok
    return {
        "status": "ok" if ok else "degraded",
        "iec2c": iec2c_ok,
        "iec2iec": iec2iec_ok,
        "lib_ok": lib_ok,
    }


@app.post("/compile", response_model=CompileResult)
def compile_st(req: CompileRequest):
    """
    Компилирует ST-код через MatIEC (iec2c).
    Возвращает результат компиляции с ошибками и предупреждениями.
    """
    return _run_matiec(req, mode="compile")


@app.post("/validate", response_model=CompileResult)
def validate_st(req: CompileRequest):
    """
    Синтаксическая проверка ST-кода (iec2iec — только парсинг).
    Быстрее чем полная компиляция.
    """
    return _run_matiec(req, mode="validate")


def _run_matiec(req: CompileRequest, mode: str) -> CompileResult:
    work = WORK_DIR / req.task_id
    work.mkdir(parents=True, exist_ok=True)

    st_file = work / "program.st"
    st_file.write_text(req.code, encoding="utf-8")

    if mode == "validate" and Path(IEC2IEC).exists():
        cmd = [IEC2IEC, "-I", str(MATIEC_LIB) + "/", str(st_file)]
    elif Path(IEC2C).exists():
        out_dir = work / "c_out"
        out_dir.mkdir(exist_ok=True)
        cmd = [IEC2C, "-I", "/usr/local/lib/matiec/", "-T", str(out_dir), str(st_file)]
    else:
        # Нет бинарей — fallback-парсер на Python
        return _python_fallback(req.code)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(work),
        )
        errors   = _extract_errors(result.stderr + result.stdout)
        warnings = _extract_warnings(result.stderr + result.stdout)
        ok = result.returncode == 0 and not errors

        return CompileResult(
            ok=ok,
            compilation_rate=1.0 if ok else 0.0,
            errors=errors,
            warnings=warnings,
            stdout=result.stdout[:4000],
            stderr=result.stderr[:4000],
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            ok=False, compilation_rate=0.0,
            errors=["Compilation timeout (30s)"],
            warnings=[], stdout="", stderr="",
        )
    except Exception as e:
        return CompileResult(
            ok=False, compilation_rate=0.0,
            errors=[str(e)],
            warnings=[], stdout="", stderr="",
        )


def _extract_errors(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines()
        if any(kw in line.lower() for kw in ["error", "syntax error", "undefined"])
    ]


def _extract_warnings(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines()
        if "warning" in line.lower()
    ]


def _python_fallback(code: str) -> CompileResult:
    """
    Базовая проверка синтаксиса без MatIEC:
    проверяем наличие обязательных ключевых слов IEC 61131-3.
    """
    errors = []
    upper = code.upper()

    has_block = any(kw in upper for kw in ["PROGRAM", "FUNCTION_BLOCK", "FUNCTION"])
    has_end   = any(kw in upper for kw in ["END_PROGRAM", "END_FUNCTION_BLOCK", "END_FUNCTION"])

    if not has_block:
        errors.append("Missing PROGRAM / FUNCTION_BLOCK / FUNCTION declaration")
    if not has_end:
        errors.append("Missing END_PROGRAM / END_FUNCTION_BLOCK / END_FUNCTION")
    if "VAR" in upper and "END_VAR" not in upper:
        errors.append("VAR section not closed with END_VAR")

    return CompileResult(
        ok=not errors,
        compilation_rate=1.0 if not errors else 0.0,
        errors=errors,
        warnings=["MatIEC binary not found — using Python fallback parser"],
        stdout="", stderr="",
    )