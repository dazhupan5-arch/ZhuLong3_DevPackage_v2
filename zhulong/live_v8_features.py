"""实机 v8 特征：从 MT5 M5 + IMF 缓存构建单行特征（v12 推理）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zhulong.training.v8.decompose import decompose_h4_to_m5
from zhulong.training.v8.features import build_v8_features
from zhulong.utils.parquet_io import read_parquet_safe
from zhulong.utils.paths import appdata_dir, install_dir, model_dir_for_symbol


def _resolve_macro_dir() -> Path:
    """优先 AppData（可写），再安装目录；支持 ZHULONG_MACRO_DIR。"""
    env = os.environ.get("ZHULONG_MACRO_DIR")
    if env:
        p = Path(env)
        if (p / "macro_daily.csv").is_file():
            return p
    for p in (appdata_dir() / "data" / "macro", install_dir() / "data" / "macro"):
        if (p / "macro_daily.csv").is_file():
            return p
    return appdata_dir() / "data" / "macro"

logger = logging.getLogger(__name__)

MIN_M5_BARS = 400


def _load_imf_cache(symbol: str) -> pd.DataFrame | None:
    csv_only = os.environ.get("ZHULONG_IMF_CSV_ONLY", "").strip() in ("1", "true", "yes")
    csv_candidates = (
        model_dir_for_symbol(symbol) / "imf_vmd.csv",
        install_dir() / "data" / "training" / "v8" / symbol / "imf_vmd.csv",
    )
    for p in csv_candidates:
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            logger.info("IMF 缓存(CSV): %s", p)
            return df.sort_index()
        except Exception as ex:
            logger.warning("IMF CSV 读取失败 %s: %s", p, ex)

    if csv_only:
        return None

    for p in (
        model_dir_for_symbol(symbol) / "imf_vmd.parquet",
        install_dir() / "data" / "training" / "v8" / symbol / "imf_vmd.parquet",
    ):
        if not p.is_file():
            continue
        df = read_parquet_safe(p)
        if df is not None:
            logger.info("IMF 缓存(Parquet): %s", p)
            return df
    return None


def m5_from_cs_bars(bars) -> pd.DataFrame:
    """从 C# 传入的 M5 列表构建 DataFrame。每项为 [time_unix, o, h, l, c, v]。"""
    if not bars:
        raise RuntimeError("C# M5 缓存为空")
    rows = []
    for b in bars:
        if isinstance(b, (list, tuple)):
            ts, o, h, lo, c, v = b[0], b[1], b[2], b[3], b[4], b[5]
        else:
            ts, o, h, lo, c, v = b["time"], b["open"], b["high"], b["low"], b["close"], b["volume"]
        rows.append((pd.to_datetime(int(ts), unit="s", utc=True).tz_localize(None), o, h, lo, c, v))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).set_index("time").sort_index()
    if len(df) < MIN_M5_BARS:
        logger.warning("C# M5 仅 %d 根（建议 ≥%d），v8 特征可能不稳定", len(df), MIN_M5_BARS)
    return df.astype(float)


def m5_from_mt5(symbol: str, bars: int = 2000) -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.terminal_info():
        if not mt5.initialize():
            raise RuntimeError(f"MT5 未连接: {mt5.last_error()}")
    broker = symbol
    if not mt5.symbol_select(broker, True):
        for alt in (f"{symbol}m", "GOLD", "XAUUSDm"):
            if mt5.symbol_select(alt, True):
                broker = alt
                break
        else:
            raise RuntimeError(f"无法选择品种 {symbol}")
    rates = mt5.copy_rates_from_pos(broker, mt5.TIMEFRAME_M5, 0, bars)
    if rates is None or len(rates) < MIN_M5_BARS:
        raise RuntimeError(f"M5 数据不足: {0 if rates is None else len(rates)}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={"tick_volume": "volume"})
    if "volume" not in df.columns:
        df["volume"] = df.get("real_volume", 1.0)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def build_live_v8_row(symbol: str, m5: pd.DataFrame | None = None) -> tuple[np.ndarray, list[str], float]:
    """返回 (feature_row, columns, atr_pct)。"""
    if m5 is None:
        m5 = m5_from_mt5(symbol)
    imf_cached = _load_imf_cache(symbol)
    if imf_cached is not None and len(imf_cached) > 0:
        imf = imf_cached.copy()
        if imf.index.tz is not None:
            imf.index = imf.index.tz_localize(None)
        imf = imf.reindex(m5.index, method="ffill")
        if imf.isna().all(axis=1).iloc[-1]:
            imf_tail = decompose_h4_to_m5(m5.tail(min(len(m5), 8000)))
            imf = imf.combine_first(imf_tail)
    else:
        logger.info("无 IMF 缓存，在线 VMD 分解（首次较慢）")
        imf = decompose_h4_to_m5(m5)

    macro_dir = _resolve_macro_dir()
    feats, cols = build_v8_features(m5, imf, macro_dir=macro_dir)
    if feats.empty:
        raise RuntimeError("v8 特征为空")
    row = feats.iloc[-1]
    atr_pct = float(row.get("atr", 0.0) / max(row.get("close", m5["close"].iloc[-1]), 1e-6) * 100.0)
    return row[cols].to_numpy(dtype=np.float32), cols, atr_pct
