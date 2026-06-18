#!/usr/bin/env python3
"""Shared V16 inference_cli runner (importlib, no subprocess deadlock)."""
from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong"
INSTALL = Path(r"C:\Program Files\ZhuLong")
BIN = _ROOT / "src" / "ZhuLong.App" / "bin" / "x64" / "Release" / "net8.0-windows10.0.19041.0" / "win-x64"

if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
except Exception:
    pass

import torch  # noqa: F401 — before onnxruntime on Windows


def data_root() -> Path:
    for base in (BIN, INSTALL, APPDATA, _ROOT):
        if (base / "models" / "horizon_v16.onnx").is_file():
            return base
    return _ROOT


def cli_script() -> tuple[Path, Path]:
    app_cli = APPDATA / "ZhuLong.PythonEngine" / "inference_cli.py"
    if app_cli.is_file():
        return app_cli, data_root()
    for base in (BIN, INSTALL, _ROOT):
        p = base / "ZhuLong.PythonEngine" / "inference_cli.py"
        if p.is_file():
            return p, base
    return _ROOT / "ZhuLong.PythonEngine" / "inference_cli.py", _ROOT


def run_cli(req: dict) -> dict:
    cli, root = cli_script()
    if not cli.is_file():
        return {"ok": False, "error": f"missing_inference_cli:{cli}"}
    req = dict(req)
    req.setdefault("root", str(root))
    req.setdefault("quick", True)
    for p in (str(APPDATA), str(root), str(INSTALL), str(BIN), str(_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(APPDATA), str(root), str(root / "ZhuLong.PythonEngine")]
    )
    spec = importlib.util.spec_from_file_location("zhulong_inference_cli", str(cli))
    if spec is None or spec.loader is None:
        return {"ok": False, "error": "cli_spec_failed"}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cmd = str(req.get("cmd", ""))
    handlers = {
        "agent_validate": getattr(mod, "_cmd_agent_validate", None),
        "agent_tick": getattr(mod, "_cmd_agent_tick", None),
        "agent_warmup": getattr(mod, "_cmd_agent_warmup", None),
        "agent_record_trade": getattr(mod, "_cmd_agent_record_trade", None),
    }
    fn = handlers.get(cmd)
    if fn is None:
        return {"ok": False, "error": f"unknown_cmd:{cmd}"}
    timeout = 90 if cmd == "agent_tick" else 30
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn, req)
        try:
            return fut.result(timeout=timeout) or {}
        except FuturesTimeout:
            return {"ok": False, "error": f"{cmd}_timeout"}
