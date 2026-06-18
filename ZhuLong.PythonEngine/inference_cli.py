#!/usr/bin/env python3
"""子进程推理 CLI — 与主进程隔离，避免 pyarrow/原生崩溃拖垮 ZhuLong.exe。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent
_install = os.environ.get("ZHULONG_INSTALL_DIR")
_appdata = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
_path_candidates: list[Path] = []
if _appdata.is_dir():
    _path_candidates.append(_appdata)
_path_candidates.append(_ENGINE)
if _install and Path(_install).is_dir():
    _path_candidates.append(Path(_install))
for p in reversed(_path_candidates):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

_ROOT = Path(_install) if _install and Path(_install).is_dir() else Path(__file__).resolve().parent.parent
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

from hotfix_loader import apply_appdata_hotfixes  # noqa: E402

apply_appdata_hotfixes()


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


_AGENT_HOTFIX_REL = (
    "zhulong/utils/json_safe.py",
    "zhulong/engine/agent_engine.py",
    "zhulong/agent/trading_agent.py",
    "zhulong/agent/horizon_predictor.py",
    "zhulong/agent/knowledge_net_kn2.py",
    "zhulong/agent/cognition.py",
    "zhulong/agent/trader_mind.py",
    "zhulong/agent/structure_service.py",
)


def _verify_agent_python_syntax(root: Path) -> None:
    """预加载前只读语法门禁；优先 AppData 热更新，不向 Program Files 写 __pycache__。"""
    import ast

    from hotfix_loader import _HOTFIX_MODULES

    appdata = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
    checked: set[str] = set()
    if appdata.is_dir():
        for rel in _HOTFIX_MODULES.values():
            path = (appdata / rel.replace("/", os.sep)).resolve()
            if path.is_file():
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                checked.add(rel.replace("\\", "/"))

    install = os.environ.get("ZHULONG_INSTALL_DIR")
    install_path = Path(install) if install and Path(install).is_dir() else root
    for rel in _AGENT_HOTFIX_REL:
        if rel.replace("\\", "/") in checked:
            continue
        path = (install_path / rel.replace("/", os.sep)).resolve()
        if path.is_file():
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _resolve_agent_config_path(cfg_rel: str, root: Path) -> Path:
    """config 优先 AppData，避免安装目录只读/旧模板。"""
    from zhulong.utils.paths import resolve_agent_config_path

    return resolve_agent_config_path(cfg_rel, root)


def _probe_horizon_v16(root: Path, config: dict) -> tuple[bool, str | None]:
    """真实加载 Horizon ONNX；失败返回可读原因（供 warmup/validate 共用）。"""
    from zhulong.agent.horizon_predictor import HorizonPredictor
    from zhulong.agent.tick_brief import StructureSnapshot

    hp = HorizonPredictor(root, config)
    if hp.is_ready:
        snap = StructureSnapshot(vector=[0.0] * 68, m5_trend=0.0)
        try:
            hp.predict(snap)
        except Exception as ex:
            return False, f"horizon_predict_failed:{type(ex).__name__}:{ex}"
        return True, None
    err = getattr(hp, "load_error", None) or "horizon_not_ready"
    kn = getattr(hp, "_kn", None)
    if kn is not None and getattr(kn, "_onnx_load_error", None):
        err = f"{err}|onnx={kn._onnx_load_error}"
    model = getattr(hp, "resolved_model_path", None)
    if model is not None:
        err = f"{err}|model={model}"
    return False, err


def _cmd_agent_warmup(req: dict) -> dict:
    """预加载 Horizon ONNX + AgentEngine（RL/KN2）；预加载前做只读 ast 语法门禁。"""
    root = Path(req.get("root") or _ROOT)
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = _resolve_agent_config_path(str(cfg_rel), root)
    if not cfg_path.is_file():
        return {"ok": False, "error": f"config_not_found:{cfg_path}"}

    config = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if not config.get("enabled", True):
        return {"ok": False, "error": "agent_disabled_in_config"}

    arch = str((config.get("architecture") or {}).get("version", "legacy"))
    sym = str(config.get("primary_symbol", "XAUUSD")).upper()
    use_rl = bool(config.get("use_rl", False))
    kn2_cfg = config.get("kn2") or {}
    kn2_on = bool(kn2_cfg.get("enabled") or kn2_cfg.get("shadow_mode"))

    horizon_ready = False
    if arch == "v16":
        ok, err = _probe_horizon_v16(root, config)
        if not ok:
            return {"ok": False, "error": err or "horizon_not_ready"}
        horizon_ready = True
    else:
        from zhulong.agent.rl_agent import resolve_knowledge_paths
        from zhulong.agent.knowledge_net import KnowledgeNetInference

        kn_path, kn_scaler = resolve_knowledge_paths(sym, config, root)
        kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
        if not kn.is_ready:
            return {"ok": False, "error": f"knowledge_not_ready:{kn_path}"}
        horizon_ready = True

    from zhulong.utils.paths import resolve_bundled_data_path

    rl_ready = False
    if use_rl:
        from zhulong.agent.rl_agent import resolve_rl_model_path

        sym = str(config.get("primary_symbol") or "XAUUSD").upper()
        rl_path = resolve_rl_model_path(sym, config, root)
        rl_zip = rl_path if rl_path.suffix.lower() == ".zip" else rl_path.with_suffix(".zip")
        if not rl_path.is_file() and not rl_zip.is_file():
            if rl_path.is_dir() and (rl_path / "policy.pth").is_file():
                rl_ready = True
            else:
                return {"ok": False, "error": f"missing_rl:{rl_path}"}
        else:
            rl_ready = True

    kn2_ready = False
    if kn2_on:
        kn2_pth = resolve_bundled_data_path(str(kn2_cfg.get("model_path", "models/kn2_trader_v16.pth")))
        if not kn2_pth.is_file():
            return {"ok": False, "error": f"missing_kn2:{kn2_pth}"}
        kn2_ready = True

    engine_preloaded = False
    if bool(req.get("preload_engine", True)):
        try:
            import sys

            _verify_agent_python_syntax(root)
            sys.stderr.write("agent_warmup: preloading AgentEngine (RL/KN2/torch)…\n")
            sys.stderr.flush()
            from zhulong.engine.agent_engine import warm_engine_cache

            warm_engine_cache(req, root)
            engine_preloaded = True
        except Exception as ex:
            return {"ok": False, "error": f"engine_preload_failed:{type(ex).__name__}:{ex}"}

    return {
        "ok": True,
        "agent": True,
        "architecture": arch,
        "horizon_ready": horizon_ready,
        "kn2_ready": kn2_ready,
        "knowledge_ready": horizon_ready,
        "rl_ready": rl_ready,
        "use_rl": use_rl,
        "engine_preloaded": engine_preloaded,
        "deferred": [] if engine_preloaded else ["rl_weights", "kn2_torch"],
    }


def _cmd_agent_record_trade(req: dict) -> dict:
    from zhulong.engine.agent_engine import record_agent_closed_trade

    root = Path(req.get("root") or _ROOT)
    sym = str(req.get("symbol") or "XAUUSD").upper()
    pnl_r = float(req.get("pnl_r") or 0.0)
    return record_agent_closed_trade(sym, pnl_r, req, root)


def _cmd_agent_record_signal(req: dict) -> dict:
    from zhulong.engine.agent_engine import record_agent_signal_emitted

    root = Path(req.get("root") or _ROOT)
    sym = str(req.get("symbol") or "XAUUSD").upper()
    return record_agent_signal_emitted(sym, req, root)


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
    cfg_path = _resolve_agent_config_path(str(cfg_rel), root)
    if not cfg_path.is_file():
        return {"ok": False, "error": f"config_not_found:{cfg_path}"}

    config = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if not config.get("enabled", True):
        return {"ok": False, "error": "agent_disabled_in_config"}

    arch = str((config.get("architecture") or {}).get("version", "legacy"))
    kn2_on = bool((config.get("kn2") or {}).get("enabled", False))
    if kn2_on and arch != "v16":
        return {"ok": False, "error": "kn2_legacy_requires_v16_architecture"}

    quick = bool(req.get("quick")) or arch == "v16"

    if quick and arch == "v16":
        from zhulong.utils.paths import resolve_bundled_data_path

        sym = str(config.get("primary_symbol", "XAUUSD")).upper()
        hp_cfg = (config.get("architecture") or {}).get("horizon_predictor") or {}
        hp_model = hp_cfg.get("model_path") or "models/horizon_v16.onnx"
        hp_scaler = hp_cfg.get("scaler_path") or "models/horizon_v16_scaler.pkl"

        hp_path = resolve_bundled_data_path(str(hp_model))
        if hp_path.suffix.lower() != ".onnx":
            hp_onnx = hp_path.with_suffix(".onnx")
            if hp_onnx.is_file():
                hp_path = hp_onnx
        if not hp_path.is_file():
            return {"ok": False, "error": f"missing_horizon:{hp_path}"}
        if hp_path.stat().st_size < 4096:
            return {"ok": False, "error": f"horizon_onnx_invalid:size={hp_path.stat().st_size}:{hp_path}"}
        hp_scaler_path = resolve_bundled_data_path(str(hp_scaler))
        if not hp_scaler_path.is_file():
            return {"ok": False, "error": f"missing_horizon_scaler:{hp_scaler_path}"}

        ok, err = _probe_horizon_v16(root, config)
        if not ok:
            return {"ok": False, "error": err or "horizon_not_ready"}

        rl_rel = (config.get("rl") or {}).get("model_path") or f"models/rl_agent_{sym[:3].lower()}"
        rl_path = resolve_bundled_data_path(str(rl_rel))
        if rl_path.suffix.lower() == ".zip":
            rl_path = rl_path.with_suffix("")
        rl_zip = Path(str(rl_path) + ".zip")
        use_rl = bool(config.get("use_rl", False))
        rl_ready = False
        if use_rl:
            if not rl_path.is_file() and not rl_zip.is_file():
                return {"ok": False, "error": f"missing_rl:{rl_path}"}
            rl_ready = True

        kn2_ready = False
        kn2_cfg = config.get("kn2") or {}
        if kn2_cfg.get("enabled") or kn2_cfg.get("shadow_mode"):
            kn2_pth = resolve_bundled_data_path(str(kn2_cfg.get("model_path", "models/kn2_trader_v16.pth")))
            if not kn2_pth.is_file():
                return {"ok": False, "error": f"missing_kn2:{kn2_pth}"}
            kn2_ready = True

        return {
            "ok": True,
            "agent": True,
            "validated": True,
            "quick": True,
            "action": "hold",
            "architecture": arch,
            "knowledge_ready": True,
            "horizon_ready": True,
            "rl_ready": rl_ready,
            "use_rl": use_rl,
            "kn2_ready": kn2_ready,
            "kn2_live": bool(kn2_cfg.get("enabled")) and not bool(kn2_cfg.get("shadow_mode")),
        }

    try:
        import onnxruntime  # noqa: F401
    except Exception as ex:
        return {"ok": False, "error": f"missing_python_module:onnxruntime:{ex}"}

    from zhulong.agent.knowledge_net import KnowledgeNetInference
    from zhulong.agent.rl_agent import resolve_knowledge_paths, resolve_rl_model_path
    from zhulong.utils.paths import resolve_bundled_data_path

    sym = str(config.get("primary_symbol", "XAUUSD")).upper()
    kn_path, kn_scaler = resolve_knowledge_paths(sym, config, root)
    if not kn_path.is_file() and not kn_path.with_suffix(".onnx").is_file():
        return {"ok": False, "error": f"missing_model:{kn_path}"}
    if kn_scaler and not Path(kn_scaler).is_file():
        return {"ok": False, "error": f"missing_scaler:{kn_scaler}"}
    kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
    if not kn.is_ready:
        return {"ok": False, "error": f"knowledge_net_not_ready:{kn_path}"}

    if quick:
        from zhulong.engine.agent_engine import run_agent_tick

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
            "symbols": [sym],
            "primary_symbol": sym,
        }
        out = run_agent_tick({sym: m5}, tick_req, root)
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
            "architecture": arch,
            "knowledge_ready": first.get("knowledge_ready"),
        }

    return {
        "ok": True,
        "agent": True,
        "validated": True,
        "quick": False,
        "architecture": arch,
        "knowledge_ready": True,
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
            "agent_warmup": _cmd_agent_warmup,
            "agent_record_trade": _cmd_agent_record_trade,
            "agent_record_signal": _cmd_agent_record_signal,
        }
        fn = handlers.get(cmd)
        payload = fn(req) if fn else {"ok": False, "error": f"unknown cmd: {cmd}"}
    except Exception as ex:
        payload = {"ok": False, "error": str(ex), "trace": traceback.format_exc()}

    from zhulong.utils.json_safe import dumps_strict

    text = dumps_strict(payload)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
