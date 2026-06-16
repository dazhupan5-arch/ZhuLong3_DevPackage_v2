#!/usr/bin/env python3
"""子进程推理 CLI — 与主进程隔离，避免 pyarrow/原生崩溃拖垮 ZhuLong.exe。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = Path(__file__).resolve().parent
for p in (_ROOT, _ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

os.environ.setdefault("ZHULONG_IMF_CSV_ONLY", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
    _install = os.environ.get("ZHULONG_INSTALL_DIR")
    if _install and os.path.isdir(_install):
        add_fn = getattr(os, "add_dll_directory", None)
        if add_fn is not None:
            try:
                add_fn(_install)
            except OSError:
                pass
        os.environ["PATH"] = _install + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# 在导入 zhulong/torch 之前预加载 onnxruntime，避免 DLL 搜索路径被覆盖
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass


def _cmd_predict(req: dict) -> dict:
    from inference import predict

    result = predict(
        req["symbol"],
        req["seq"],
        req["hourly"],
        req["macro"],
        req.get("m5_bars"),
    )
    return {"ok": True, "result": result}


def _cmd_warmup(req: dict) -> dict:
    from inference import warmup

    return {"ok": True, **warmup(req.get("symbols") or [])}


def _cmd_validate(req: dict) -> dict:
    from inference import validate_models

    return {"ok": True, **validate_models(req.get("symbols") or [])}


def _bars_to_df(bars: list) -> "pd.DataFrame":
    import pandas as pd

    from zhulong.utils.time_index import normalize_m5_index

    rows = []
    for b in bars:
        ts = pd.Timestamp(int(b[0]), unit="s", tz="UTC")
        rows.append(
            {
                "time": ts,
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": float(b[5]) if len(b) > 5 else 0.0,
            }
        )
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return normalize_m5_index(df[["open", "high", "low", "close", "volume"]])


def _cmd_multi_strategy_tick(req: dict) -> dict:
    from zhulong.engine.multi_strategy_engine import MultiStrategyEngine, load_multi_strategy_config
    from zhulong.engine.scheduler_engine import SchedulerEngine, merge_scheduler_config
    from zhulong.engine.runtime_config import apply_runtime_primary, bind_engine_primary

    root = Path(req.get("root") or _ROOT)
    cfg_rel = req.get("config_path") or "config/config_multi_strategy.json"
    cfg_path = Path(cfg_rel)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    config = load_multi_strategy_config(cfg_path)
    sched_path = (config.get("scheduler") or {}).get("config_path") or "config/config_scheduler.json"
    config = merge_scheduler_config(config, sched_path)

    runtime_primary = apply_runtime_primary(config, req.get("primary_symbol"))

    m5_by_symbol: dict = {}
    for sym, bars in (req.get("m5_bars_by_symbol") or {}).items():
        if bars:
            m5_by_symbol[sym] = _bars_to_df(bars)

    symbols = req.get("symbols") or list(m5_by_symbol.keys())
    use_scheduler = bool(config.get("scheduler_enabled"))
    if use_scheduler:
        engine = SchedulerEngine(config, root=root)
    else:
        engine = MultiStrategyEngine(config, root=root)
    engine.macro_silence = bool(req.get("macro_silence", False))
    if runtime_primary:
        bind_engine_primary(engine, runtime_primary)

    results = engine.tick_symbols(m5_by_symbol, symbols)
    return {"ok": True, "scheduler": use_scheduler, "primary_symbol": runtime_primary, "results": results}


def _cmd_agent_tick(req: dict) -> dict:
    from zhulong.engine.agent_engine import run_agent_tick

    root = Path(req.get("root") or _ROOT)
    m5_by_symbol: dict = {}
    for sym, bars in (req.get("m5_bars_by_symbol") or {}).items():
        if bars:
            m5_by_symbol[sym] = _bars_to_df(bars)
    return run_agent_tick(m5_by_symbol, req, root)


def _cmd_agent_validate(req: dict) -> dict:
    import importlib

    for mod in ("numpy", "pandas", "sklearn", "joblib"):
        try:
            importlib.import_module(mod)
        except Exception as ex:
            return {"ok": False, "error": f"missing_python_module:{mod}:{ex}"}

    root = Path(req.get("root") or _ROOT)
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = Path(cfg_rel)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.is_file():
        return {"ok": False, "error": f"config_not_found:{cfg_path}"}

    from zhulong.engine.agent_engine import load_agent_config, run_agent_tick

    config = load_agent_config(cfg_path, root=root)
    if not config.get("enabled", True):
        return {"ok": False, "error": "agent_disabled_in_config"}

    if bool((config.get("kn2") or {}).get("enabled", False)):
        return {"ok": False, "error": "kn2_removed_use_architecture_v16"}

    kn_ready = False
    try:
        import onnxruntime  # noqa: F401
    except Exception as ex:
        return {"ok": False, "error": f"missing_python_module:onnxruntime:{ex}"}
    from zhulong.agent.knowledge_net import KnowledgeNetInference
    from zhulong.agent.rl_agent import resolve_knowledge_paths

    kn_path, kn_scaler = resolve_knowledge_paths(
        str(config.get("primary_symbol", "XAUUSD")), config, root
    )
    if not kn_path.is_file() and not kn_path.with_suffix(".onnx").is_file():
        return {"ok": False, "error": f"missing_model:{kn_path}"}
    if kn_scaler and not Path(kn_scaler).is_file():
        return {"ok": False, "error": f"missing_scaler:{kn_scaler}"}
    kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
    if not kn.is_ready:
        return {"ok": False, "error": f"knowledge_net_not_ready:{kn_path}"}
    kn_ready = True

    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=120, freq="5min", tz="UTC")
    close = 2400 + np.cumsum(np.random.randn(120) * 0.1)
    m5 = pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 50.0},
        index=idx,
    )
    tick_req = {
        "config_path": str(cfg_path),
        "symbols": ["XAUUSD"],
        "primary_symbol": "XAUUSD",
    }
    out = run_agent_tick({"XAUUSD": m5}, tick_req, root)
    if not out.get("ok"):
        return out
    if not out.get("agent"):
        return {"ok": False, "error": out.get("reason") or "agent_disabled"}
    results = out.get("results") or []
    if not results:
        return {"ok": False, "error": "agent_empty_results"}
    first = results[0]
    if first.get("skipped"):
        return {"ok": False, "error": f"agent_skipped:{first.get('reason')}"}
    return {
        "ok": True,
        "agent": True,
        "validated": True,
        "action": first.get("action"),
        "architecture": str((config.get("architecture") or {}).get("version", "legacy")),
        "knowledge_ready": first.get("knowledge_ready"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON request file")
    parser.add_argument("--output", help="JSON response file (default stdout)")
    args = parser.parse_args()

    try:
        req = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        cmd = req.get("cmd", "predict")
        handlers = {
            "predict": _cmd_predict,
            "warmup": _cmd_warmup,
            "validate": _cmd_validate,
            "multi_strategy_tick": _cmd_multi_strategy_tick,
            "agent_tick": _cmd_agent_tick,
            "agent_validate": _cmd_agent_validate,
        }
        fn = handlers.get(cmd)
        payload = fn(req) if fn else {"ok": False, "error": f"unknown cmd: {cmd}"}
    except Exception as ex:
        payload = {"ok": False, "error": str(ex), "trace": traceback.format_exc()}

    text = json.dumps(payload, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
