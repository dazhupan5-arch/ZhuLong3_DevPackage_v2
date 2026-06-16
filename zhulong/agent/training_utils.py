"""训练流水线共享工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from zhulong.strategies.indicators import atr_series

ROOT = Path(__file__).resolve().parent.parent.parent

SYMBOL_DEFAULTS = {
    "XAUUSD": {
        "csv_candidates": [
            "data/training/lgb/XAUUSD/XAUUSD_M5.csv",
            "data/training/XAUUSD_M5.csv",
            "data/XAUUSD_M5.csv",
        ],
        "npz": "data/training_data.npz",
        "knowledge_model": "models/knowledge_net.pth",
        "knowledge_scaler": "models/knowledge_scaler.pkl",
        "rl_model": "models/rl_agent_xau",
        "point_cost": 0.2,
    },
    "USOIL": {
        "csv_candidates": [
            "data/training/USOIL_M5.csv",
            "data/USOIL_M5.csv",
            "data/training/lgb/USOIL/USOIL_M5.csv",
        ],
        "npz": "data/oil_training_data.npz",
        "knowledge_model": "models/knowledge_net_oil.pth",
        "knowledge_scaler": "models/knowledge_scaler_oil.pkl",
        "rl_model": "models/rl_agent_oil",
        "point_cost": 0.03,
    },
}


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "config_training.yaml"
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve_symbol_paths(symbol: str, cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    sym = symbol.strip().upper()
    defaults = SYMBOL_DEFAULTS.get(sym, SYMBOL_DEFAULTS["XAUUSD"]).copy()
    cfg = cfg or {}
    data_cfg = cfg.get("data") or {}

    npz_key = "xau_path" if sym == "XAUUSD" else "oil_path"
    npz_rel = data_cfg.get(npz_key, defaults["npz"])

    kn_cfg = cfg.get("knowledge_net") or {}
    rl_cfg = cfg.get("rl") or {}
    env_cfg = cfg.get("env") or {}

    kn_model = kn_cfg.get(f"model_{sym.lower()}") or defaults["knowledge_model"]
    if sym == "USOIL" and "model_oil" in kn_cfg:
        kn_model = kn_cfg["model_oil"]
    rl_model = rl_cfg.get(f"model_{sym.lower()}") or defaults["rl_model"]
    if sym == "USOIL" and "model_oil" in rl_cfg:
        rl_model = rl_cfg["model_oil"]

    return {
        "symbol": sym,
        "npz": ROOT / npz_rel,
        "knowledge_model": ROOT / kn_model,
        "knowledge_scaler": ROOT / defaults["knowledge_scaler"],
        "rl_model": ROOT / rl_model,
        "point_cost": float((env_cfg.get("point_cost") or {}).get(sym, defaults["point_cost"])),
        "csv": find_csv(sym, _csv_override(sym, data_cfg)),
    }


def _csv_override(symbol: str, data_cfg: dict[str, Any]) -> str | None:
    sym = symbol.upper()
    if sym == "XAUUSD":
        return data_cfg.get("xau_csv") or data_cfg.get("csv")
    if sym == "USOIL":
        return data_cfg.get("oil_csv") or data_cfg.get("csv")
    return data_cfg.get("csv")


def find_csv(symbol: str, override: str | None = None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    sym = symbol.upper()
    for rel in SYMBOL_DEFAULTS.get(sym, SYMBOL_DEFAULTS["XAUUSD"])["csv_candidates"]:
        p = ROOT / rel
        if p.is_file():
            return p
    return ROOT / SYMBOL_DEFAULTS.get(sym, SYMBOL_DEFAULTS["XAUUSD"])["csv_candidates"][0]


def clean_m5_bars(
    df: pd.DataFrame,
    *,
    spike_atr_mult: float = 10.0,
    max_body_pct: float = 0.025,
    drop_zero_volume: bool = True,
    report: dict[str, int] | None = None,
) -> pd.DataFrame:
    """清洗 M5 OHLCV：去重、OHLC 合法性、异常 spike、零成交量等。"""
    r: dict[str, int] = report if report is not None else {}
    n0 = len(df)
    out = df.sort_values("time").reset_index(drop=True)

    def _drop(mask: pd.Series, key: str) -> None:
        nonlocal out
        n_bad = int((~mask).sum())
        if n_bad:
            r[key] = r.get(key, 0) + n_bad
            out = out.loc[mask].reset_index(drop=True)

    cols = ["open", "high", "low", "close", "volume"]
    _drop(out[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1), "non_finite")
    for c in ("open", "high", "low", "close"):
        _drop(out[c] > 0, f"non_positive_{c}")

    hi_ref = out[["open", "close", "high"]].max(axis=1)
    lo_ref = out[["open", "close", "low"]].min(axis=1)
    _drop((out["high"] >= hi_ref - 1e-9) & (out["low"] <= lo_ref + 1e-9) & (out["high"] >= out["low"]), "ohlc_invalid")

    dup = out["time"].duplicated(keep="last")
    if dup.any():
        r["duplicate_time"] = int(dup.sum())
        out = out.loc[~dup].reset_index(drop=True)

    if drop_zero_volume:
        _drop(out["volume"] > 0, "zero_volume")

    close = out["close"].astype(np.float64)
    prev_close = close.shift(1)
    tr = np.maximum(out["high"].values - out["low"].values, np.abs(out["high"].values - prev_close.fillna(out["open"]).values))
    tr = np.maximum(tr, np.abs(out["low"].values - prev_close.fillna(out["open"]).values))
    atr = pd.Series(tr, index=out.index).rolling(14, min_periods=1).mean().values
    bar_range = (out["high"] - out["low"]).values
    body_pct = (np.abs(out["close"] - out["open"]) / np.maximum(close, 1e-9)).values
    spike = (bar_range > spike_atr_mult * np.maximum(atr, close.values * 1e-4)) | (body_pct > max_body_pct)
    _drop(~spike, "price_spike")

    # 单根 bad tick：收盘价相对前收跳变 >1.5% 且下一根 revert >50%
    if len(out) >= 3:
        ret1 = (close - prev_close) / np.maximum(prev_close, 1e-9)
        next_ret = (close.shift(-1) - close) / np.maximum(close, 1e-9)
        bad_tick = (ret1.abs() > 0.015) & (ret1 * next_ret < 0) & (next_ret.abs() > ret1.abs() * 0.5)
        bad_tick = bad_tick.fillna(False)
        _drop(~bad_tick, "bad_tick_revert")

    r["rows_in"] = n0
    r["rows_out"] = len(out)
    r["rows_removed"] = n0 - len(out)
    return out[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def load_m5_csv(
    path: Path,
    start: str | None = None,
    end: str | None = None,
    *,
    clean: bool = True,
    filter_weekend: bool = True,
    filter_low_volume: bool = True,
    clean_report: dict[str, int] | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] >= 7 and not _has_named_columns(path):
        raw.columns = ["date", "time_str", "open", "high", "low", "close", "volume"][: raw.shape[1]]
        ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time_str"].astype(str), utc=True)
        df = raw.copy()
        df["time"] = ts
    else:
        df = pd.read_csv(path)
        if "time" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "time"})
        if "time" not in df.columns:
            raise ValueError(f"CSV 缺少 time 列: {path}")
        df["time"] = pd.to_datetime(df["time"], utc=True)

    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    df = df.sort_values("time").reset_index(drop=True)
    if start:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["time"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if filter_weekend:
        df = df[df["time"].dt.dayofweek < 5].copy()
    if filter_low_volume and vol_col in df.columns and len(df) > 100:
        q = df[vol_col].quantile(0.05)
        n_low_vol = int((df[vol_col] < q).sum())
        if clean_report is not None and n_low_vol:
            clean_report["low_volume_bottom5pct"] = n_low_vol
        df = df[df[vol_col] >= q].copy()
    if vol_col != "volume":
        df = df.rename(columns={vol_col: "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    if clean:
        df = clean_m5_bars(df, report=clean_report)
    return df


def _has_named_columns(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    if not head:
        return False
    first = head[0].lower()
    return "time" in first or "open" in first


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    ohlc = df.set_index("time")[["high", "low", "close"]]
    atr = atr_series(ohlc, period).bfill().fillna(ohlc["close"] * 0.001)
    return atr.values.astype(np.float64)


def build_signed_labels(close: np.ndarray, horizon: int = 12, thr: float = 0.002) -> np.ndarray:
    """标签：-1 空 / 0 观望 / 1 多。"""
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - horizon):
        ret = (close[i + horizon] - close[i]) / max(close[i], 1e-9)
        if ret > thr:
            labels[i] = 1
        elif ret < -thr:
            labels[i] = -1
    return labels


def signed_to_class(labels: np.ndarray) -> np.ndarray:
    """-1,0,1 → 0,1,2（short, flat, long）。"""
    return (labels.astype(np.int64) + 1).clip(0, 2)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def filter_npz_by_year(data: dict[str, np.ndarray], year: int) -> dict[str, np.ndarray]:
    times = pd.to_datetime(data["time"])
    mask = times.year == year
    n = len(mask)
    out: dict[str, np.ndarray] = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.shape[0] == n:
            out[k] = arr[mask]
        elif k == "symbol":
            out[k] = arr
    return out


def ensure_logs_dir() -> Path:
    p = ROOT / "logs" / "training"
    p.mkdir(parents=True, exist_ok=True)
    return p
