"""实机 USOIL v1 特征：MT5 M5 + IMF 缓存 → 121 维单行特征。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zhulong.live_v8_features import _resolve_macro_dir, m5_from_mt5
from zhulong.training.oil_v1.features import build_oil_v1_features
from zhulong.training.v8.decompose import decompose_h4_to_m5
from zhulong.utils.paths import install_dir, model_dir_for_symbol

logger = logging.getLogger(__name__)

MIN_M5_BARS = 400
OIL_ALIASES = ("USOIL", "XTIUSD", "WTI", "CL-OIL", "USOILm", "XTIUSDm")


def _load_imf_cache(symbol: str, imf_cache: Path | None) -> pd.DataFrame | None:
    candidates = []
    if imf_cache:
        candidates.append(imf_cache)
    candidates.extend([
        model_dir_for_symbol(symbol) / "imf_vmd.parquet",
        model_dir_for_symbol(symbol) / "v1" / "imf_vmd.parquet",
        install_dir() / "data" / "training" / "oil_v1" / symbol / "imf_vmd.parquet",
    ])
    for p in candidates:
        if p.is_file():
            return pd.read_parquet(p)
    return None


def m5_from_mt5_oil(broker_symbol: str, bars: int = 2000) -> pd.DataFrame:
    """拉取原油 M5，自动尝试经纪商符号别名。"""
    import MetaTrader5 as mt5

    if not mt5.terminal_info():
        if not mt5.initialize():
            raise RuntimeError(f"MT5 未连接: {mt5.last_error()}")
    tried = [broker_symbol]
    for alt in OIL_ALIASES:
        if alt not in tried:
            tried.append(alt)
    broker = None
    for sym in tried:
        if mt5.symbol_select(sym, True):
            broker = sym
            break
    if broker is None:
        raise RuntimeError(f"无法选择原油品种，已尝试: {tried}")
    rates = mt5.copy_rates_from_pos(broker, mt5.TIMEFRAME_M5, 0, bars)
    if rates is None or len(rates) < MIN_M5_BARS:
        raise RuntimeError(f"原油 M5 不足: {0 if rates is None else len(rates)} ({broker})")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={"tick_volume": "volume"})
    if "volume" not in df.columns:
        df["volume"] = df.get("real_volume", 1.0)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def build_live_oil_row(
    symbol: str = "USOIL",
    m5: pd.DataFrame | None = None,
    broker_symbol: str | None = None,
    imf_cache: Path | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    if m5 is None:
        m5 = m5_from_mt5_oil(broker_symbol or symbol)
    imf_cached = _load_imf_cache(symbol, imf_cache)
    if imf_cached is not None and len(imf_cached) > 0:
        imf = imf_cached.copy()
        if imf.index.tz is not None:
            imf.index = imf.index.tz_localize(None)
        imf = imf.reindex(m5.index, method="ffill")
        if imf.isna().all(axis=1).iloc[-1]:
            imf_tail = decompose_h4_to_m5(m5.tail(min(len(m5), 8000)))
            imf = imf.combine_first(imf_tail)
    else:
        logger.info("无原油 IMF 缓存，在线 VMD（首次较慢）")
        imf = decompose_h4_to_m5(m5)
    feats, cols = build_oil_v1_features(m5, imf, macro_dir=_resolve_macro_dir())
    if feats.empty:
        raise RuntimeError("oil v1 特征为空")
    t = feats.index[-1]
    row = feats.loc[t, cols].to_numpy(dtype=np.float32)
    return row, cols, m5, feats.loc[[t], cols]
