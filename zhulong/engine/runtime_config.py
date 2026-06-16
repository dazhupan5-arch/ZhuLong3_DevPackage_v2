"""实机运行时覆盖（WinUI 设置页主品种等）。"""

from __future__ import annotations

from typing import Any


def apply_runtime_primary(config: dict[str, Any], primary: str | None) -> str | None:
    """将 C# / 用户 config 的主品种写入多策略/调度配置。"""
    if not primary or not str(primary).strip():
        sm = config.get("state_machine") or {}
        return sm.get("primary_symbol")
    sym = str(primary).strip().upper()
    config.setdefault("state_machine", {})["primary_symbol"] = sym
    sched_core = config.get("scheduler_core")
    if isinstance(sched_core, dict):
        sched_core.setdefault("state_machine", {})["primary_symbol"] = sym
    return sym


def bind_engine_primary(engine: Any, primary: str) -> None:
    sym = primary.strip().upper()
    if hasattr(engine, "set_primary_symbol"):
        engine.set_primary_symbol(sym)
        return
    if hasattr(engine, "state_machine"):
        engine.state_machine.primary_symbol = sym
    if hasattr(engine, "scheduler"):
        engine.scheduler.primary_symbol = sym
        engine.scheduler.state_machine.primary_symbol = sym
    if hasattr(engine, "primary_symbol"):
        engine.primary_symbol = sym
