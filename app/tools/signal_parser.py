"""
app/tools/signal_parser.py — Парсинг таблицы сигналов CSV/XLSX → ST VAR секция.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


SIGNAL_TYPE_MAP = {
    "DI": ("BOOL",  "%IX"),
    "DO": ("BOOL",  "%QX"),
    "AI": ("REAL",  "%IW"),
    "AO": ("REAL",  "%QW"),
}

ELBRUS_STEP = {"AI": 2, "AO": 2}  # Эльбрус: AI/AO адреса через 2


def parse_signal_table(path: str) -> list[dict]:
    """Разбирает CSV/XLSX в список словарей сигналов."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    df.columns = [c.strip() for c in df.columns]
    signals = []
    for _, row in df.iterrows():
        signals.append({
            "name":    str(row.get("SignalName", row.get("Name", "SIG"))).strip(),
            "type":    str(row.get("Type", "DI")).strip().upper(),
            "address": str(row.get("Address", "")).strip(),
            "range":   str(row.get("Range", "")).strip(),
            "unit":    str(row.get("Engineering_Unit", "")).strip(),
            "desc":    str(row.get("Description", "")).strip(),
        })
    return signals


def signals_to_st_var(signals: list[dict], controller: str = "elbrus") -> str:
    """Генерирует VAR-секцию ST из списка сигналов."""
    lines = ["VAR"]
    ai_addr = 0
    ao_addr = 0

    for s in signals:
        sig_type = s["type"]
        name     = s["name"]
        addr     = s["address"]
        desc     = s["desc"] or name
        unit     = s["unit"]
        comment  = f"{desc}" + (f" ({unit})" if unit else "")

        dt, _ = SIGNAL_TYPE_MAP.get(sig_type, ("BOOL", "%IX"))

        if sig_type == "AI":
            if not addr:
                addr = f"%IW{ai_addr}"
                ai_addr += ELBRUS_STEP.get("AI", 1)
            lines.append(f"    {name}_RAW : INT;  (* {addr} — {comment} RAW *)")
            lines.append(f"    {name}     : REAL; (* Scaled value *)")
        elif sig_type == "AO":
            if not addr:
                addr = f"%QW{ao_addr}"
                ao_addr += ELBRUS_STEP.get("AO", 1)
            lines.append(f"    {name}     : REAL; (* {addr} — {comment} *)")
            lines.append(f"    {name}_OUT : INT;  (* Scaled for AO *)")
        else:
            lines.append(f"    {name} : {dt}; (* {addr} — {comment} *)")

    lines.append("    (* Internal *)")
    lines.append("    SCALE_FACTOR : REAL := 1.0 / 32767.0;")
    lines.append("END_VAR")
    return "\n".join(lines)


def signals_to_scale_body(signals: list[dict]) -> str:
    """Генерирует тело программы: масштабирование AI/AO."""
    lines = ["(* Scaling *)", ""]
    for s in signals:
        if s["type"] == "AI":
            rng = s.get("range", "0-100")
            try:
                parts = rng.replace("..", "-").split("-")
                lo, hi = float(parts[0]), float(parts[-1])
                span = hi - lo
            except Exception:
                lo, span = 0.0, 100.0
            lines.append(
                f"{s['name']} := REAL({s['name']}_RAW) * ({span} / 32767.0) + {lo};"
            )
        elif s["type"] == "AO":
            rng = s.get("range", "0-100")
            try:
                parts = rng.replace("..", "-").split("-")
                lo, hi = float(parts[0]), float(parts[-1])
                span = hi - lo
            except Exception:
                lo, span = 0.0, 100.0
            lines.append(
                f"{s['name']}_OUT := REAL_TO_INT({s['name']} * (32767.0 / {span}));"
            )
    return "\n".join(lines)