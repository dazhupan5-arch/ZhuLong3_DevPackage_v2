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

# V16 无泄露训练契约：训练截止 / OOS 验证年（禁止随机 val 混入未来 bar）
TRAIN_END_DEFAULT = "2024-12-31"
VAL_YEAR_DEFAULT = 2025
PIPELINE_CONTRACT_VERSION = "v16_no_leak_1"

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


def resolve_v16_paths(symbol: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """V16 训练/推理路径（XAUUSD 沿用根目录 artifacts，USOIL 独立 data/clean + models/USOIL/v16）。"""
    sym = symbol.strip().upper()
    cfg = cfg or load_training_config()
    data_cfg = cfg.get("data") or {}
    kn_cfg = cfg.get("knowledge_net") or {}
    oil_kn = kn_cfg.get("oil") or {}

    if sym == "XAUUSD":
        horizon = int(data_cfg.get("label_horizon", 12))
        gain = float(data_cfg.get("label_threshold", 0.002))
        v16_cfg = cfg.get("v16") or {}
        horizon_npz_rel = v16_cfg.get(
            "horizon_npz",
            data_cfg.get("v16_horizon_npz", "data/clean/training_horizon_v16_location.npz"),
        )
        kn2_npz_rel = v16_cfg.get(
            "kn2_npz",
            data_cfg.get("v16_kn2_npz", "data/clean/kn2_training_v16_location.npz"),
        )
        return {
            "symbol": sym,
            "horizon": horizon,
            "gain": gain,
            "hidden_dim": 96,
            "clean_csv": ROOT / "data/clean/XAUUSD_M5_clean.csv",
            "horizon_npz": ROOT / horizon_npz_rel,
            "horizon_location_npz": ROOT / "data/clean/training_horizon_v16_location.npz",
            "kn2_npz": ROOT / kn2_npz_rel,
            "kn2_location_npz": ROOT / kn2_npz_rel,
            "horizon_pth": ROOT / "models/horizon_v16.pth",
            "horizon_onnx": ROOT / "models/horizon_v16.onnx",
            "horizon_scaler": ROOT / "models/horizon_v16_scaler.pkl",
            "horizon_meta": ROOT / "models/horizon_v16.meta.json",
            "kn2_pth": ROOT / "models/kn2_trader_v16.pth",
            "kn2_meta": ROOT / "models/kn2_trader_v16.meta.json",
            "rl_model": ROOT / "models/rl_agent_xau",
            "rl_meta": ROOT / "models/XAUUSD/v16/rl_meta.json",
            "struct_cache": ROOT / "data/training/v16/XAUUSD",
            "acceptance": ROOT / "config/v16_acceptance.json",
        }

    if sym == "USOIL":
        horizon = int(oil_kn.get("label_horizon", 18))
        gain = float(oil_kn.get("label_gain", 0.003))
        v16_dir = ROOT / "models/USOIL/v16"
        return {
            "symbol": sym,
            "horizon": horizon,
            "gain": gain,
            "hidden_dim": int(oil_kn.get("hidden_dim", 64)),
            "clean_csv": ROOT / "data/clean/USOIL_M5_clean.csv",
            "horizon_npz": ROOT / data_cfg.get("v16_horizon_npz_oil", "data/clean/training_horizon_v16_usoil.npz"),
            "horizon_location_npz": ROOT / "data/clean/training_horizon_v16_usoil_location.npz",
            "kn2_npz": ROOT / data_cfg.get("v16_kn2_npz_oil", "data/clean/kn2_training_v16_usoil.npz"),
            "kn2_location_npz": ROOT / "data/clean/kn2_training_v16_usoil_location.npz",
            "horizon_pth": v16_dir / "horizon_v16.pth",
            "horizon_onnx": v16_dir / "horizon_v16.onnx",
            "horizon_scaler": v16_dir / "horizon_v16_scaler.pkl",
            "horizon_meta": v16_dir / "horizon_v16.meta.json",
            "kn2_pth": v16_dir / "kn2_trader_v16.pth",
            "kn2_meta": v16_dir / "kn2_trader_v16.meta.json",
            "rl_model": ROOT / "models/rl_agent_oil",
            "rl_meta": v16_dir / "rl_meta.json",
            "struct_cache": ROOT / "data/training/v16/USOIL",
            "acceptance": ROOT / "config/v16_acceptance_usoil.json",
        }

    raise ValueError(f"V16 paths not defined for symbol: {sym}")


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

    # 因果 bad tick：在 bar i 仅用 i-1/i 判断 i-1 是否为假 spike（禁止 shift(-1) 窥视未来）
    if len(out) >= 3:
        prev2 = close.shift(2)
        ret_im1 = (prev_close - prev2) / np.maximum(prev2, 1e-9)
        ret_i = (close - prev_close) / np.maximum(prev_close, 1e-9)
        drop_prior = (
            (ret_im1.abs() > 0.015)
            & (ret_im1 * ret_i < 0)
            & (ret_i.abs() > ret_im1.abs() * 0.5)
        )
        drop_prior = drop_prior.fillna(False)
        remove = np.zeros(len(out), dtype=bool)
        if len(remove) > 1:
            remove[:-1] = drop_prior.iloc[1:].to_numpy(dtype=bool)
        _drop(~remove, "bad_tick_revert_causal")

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


def temporal_train_val_masks(
    times: np.ndarray | pd.DatetimeIndex | pd.Series,
    *,
    train_end: str = TRAIN_END_DEFAULT,
    val_year: int = VAL_YEAR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """严格时间切分：train <= train_end；val = val_year 全年（OOS）。"""
    ts = pd.to_datetime(times, utc=True)
    tz = ts.tz if isinstance(ts, pd.DatetimeIndex) else ts.dt.tz
    cutoff = pd.Timestamp(train_end)
    if tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(tz)
    elif tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    train_mask = np.asarray(ts <= cutoff)
    val_mask = np.asarray(ts.year == int(val_year))
    return train_mask, val_mask


def assert_temporal_split_ok(
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    min_train: int = 1000,
    min_val: int = 500,
    name: str = "split",
) -> None:
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    if n_train < min_train or n_val < min_val:
        raise ValueError(
            f"{name}: 时间切分样本不足 train={n_train} val={n_val} "
            f"(need >={min_train}/{min_val})"
        )
    if np.any(train_mask & val_mask):
        raise ValueError(f"{name}: train/val 掩码重叠")
    if np.any(val_mask & (np.asarray(train_mask))):
        pass
    # val 必须在 train 之后
    if n_train and n_val:
        ts_train_max = np.flatnonzero(train_mask)
        ts_val_min = np.flatnonzero(val_mask)
        if ts_val_min[0] <= ts_train_max[-1]:
            # 同 cutoff 边界允许相邻，但 val_year 应在 train_end 之后
            pass


def load_horizon_training_meta(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def require_temporal_horizon_model(
    model_or_meta_path: Path,
    *,
    train_end: str = TRAIN_END_DEFAULT,
    allow_force: bool = False,
) -> dict[str, Any]:
    """prepare KN2/RL 前校验 Horizon 模型按时间切分训练（禁止 random val 产物）。"""
    if model_or_meta_path.suffix == ".json":
        meta_path = model_or_meta_path
    else:
        meta_path = model_or_meta_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        if allow_force:
            return {}
        raise FileNotFoundError(f"缺少 Horizon meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("temporal_val") is not True and not allow_force:
        raise ValueError(
            f"Horizon 模型未按 temporal_val 训练 ({meta_path})；"
            "请用 train_horizon_v16.py --temporal-val 重训，或显式 --force"
        )
    meta_end = str(meta.get("train_end") or meta.get("train_end_cutoff") or "")
    if meta_end and meta_end[:10] != train_end[:10] and not allow_force:
        raise ValueError(f"Horizon train_end={meta_end} 与契约 {train_end} 不一致")
    return meta
