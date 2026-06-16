"""
烛龙 Python 推理模块 — 供 C# Python.NET 调用。
接口: predict(symbol, seq, hourly, macro) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许从 ZhuLong.PythonEngine 或开发根目录加载 zhulong 包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

_engine_cache: dict = {}


def _get_engine(symbol: str):
    if symbol not in _engine_cache:
        from zhulong.inference_engine import InferenceEngine
        from zhulong.config_loader import load_config

        cfg = load_config().get("model", default={}) or {}
        eng = InferenceEngine(cfg)
        eng.load(symbol)
        _engine_cache[symbol] = eng
    return _engine_cache[symbol]


def predict(symbol: str, seq, hourly, macro, m5_bars=None):
    """
    Parameters
    ----------
    seq : list[list[float]] 或 numpy (60, F)
    hourly : list[float] 长度 10
    macro : list[float] 长度 8
    m5_bars : optional list[[unix_ts, o, h, l, c, v], ...] 由 C# FeatureCache 传入
    """
    engine = _get_engine(symbol)
    seq_arr = np.asarray(seq, dtype=np.float32)
    hourly_arr = np.asarray(hourly, dtype=np.float32)
    macro_arr = np.asarray(macro, dtype=np.float32)
    m5_df = None
    if m5_bars is not None and len(m5_bars) > 0:
        from zhulong.live_v8_features import m5_from_cs_bars

        m5_df = m5_from_cs_bars(m5_bars)
    result = engine.predict(symbol, seq_arr, hourly_arr, macro_arr, m5=m5_df)
    return {
        "direction": int(result["direction"]),
        "confidence": float(result["confidence"]),
        "entry_offset": float(result["entry_offset"]),
        "expected_return": float(result["expected_return"]),
    }


def validate_models(symbols):
    from zhulong.inference_engine import InferenceEngine
    from zhulong.config_loader import load_config

    cfg = load_config().get("model", default={}) or {}
    eng = InferenceEngine(cfg)
    missing = [s for s in symbols if not eng.validate_symbol_models(s)]
    return {"ok": len(missing) == 0, "missing": missing}


def validate_production_models(symbols):
    """仅当 manifest acceptance_passed=true 且 kind!=demo 时视为可推理。"""
    from zhulong.utils.paths import model_dir_for_symbol
    import json

    ready, pending = [], []
    for sym in symbols:
        mp = model_dir_for_symbol(sym) / "manifest.json"
        if not mp.is_file():
            pending.append({"symbol": sym, "reason": "missing_manifest"})
            continue
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as ex:
            pending.append({"symbol": sym, "reason": f"bad_manifest:{ex}"})
            continue
        if manifest.get("kind") == "demo":
            pending.append({"symbol": sym, "reason": "demo_only"})
            continue
        if not manifest.get("acceptance_passed"):
            pending.append({"symbol": sym, "reason": "not_accepted"})
            continue
        ready.append(sym)
    return {"ok": len(pending) == 0 and len(ready) == len(symbols), "ready": ready, "pending": pending}


def warmup(symbols):
    """预加载模型制品（仅 load，不做完整 predict）。"""
    warmed = []
    failed = []
    for sym in symbols:
        try:
            _get_engine(sym)
            warmed.append(sym)
        except Exception as ex:
            failed.append({"symbol": sym, "error": str(ex)})
    return {"ok": True, "warmed": warmed, "failed": failed}
