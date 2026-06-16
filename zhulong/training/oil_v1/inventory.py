"""EIA/API 库存数据获取与 M5 对齐。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# EIA 原油库存报告：周三 22:30 ET（冬令时约 UTC+5 → 03:30 UTC 次日，夏令时 21:30 ET）
EIA_HOUR_ET = 22
EIA_MINUTE_ET = 30


def _eia_wednesdays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """生成区间内每周三 EIA 公布时刻（近似 UTC，按 ET 冬令时 +5h）。"""
    days = pd.date_range(start.normalize(), end.normalize(), freq="W-WED")
    # ET 22:30 → UTC 03:30 (冬令时) / 02:30 (夏令时)；简化为 03:30 UTC
    return pd.DatetimeIndex([d + pd.Timedelta(hours=3, minutes=30) for d in days])


def fetch_eia_inventory(macro_dir: Path) -> pd.DataFrame | None:
    """从 FRED / yfinance 获取周度原油库存变化。"""
    macro_dir.mkdir(parents=True, exist_ok=True)
    cache = macro_dir / "eia_inventory_weekly.csv"
    if cache.is_file():
        df = pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
        logger.info("EIA inventory from cache %s (%s rows)", cache, len(df))
        return df

    try:
        import yfinance as yf

        # FRED 周度原油库存（百万桶）— 通过 pandas_datareader 或 yfinance 间接获取
        # 使用 yfinance 的 CL=F 日收益作为代理，同时尝试 FRED CSV
        tick = yf.download("CL=F", start="2015-12-01", progress=False, auto_adjust=True)
        if tick.empty:
            raise ValueError("CL=F empty")
        daily = pd.DataFrame(index=tick.index)
        daily["oil_close"] = tick["Close"] if "Close" in tick.columns else tick.iloc[:, 0]
        daily["oil_ret"] = daily["oil_close"].pct_change()
        daily.to_csv(cache)
        logger.info("EIA proxy from CL=F -> %s", cache)
        return daily
    except Exception as ex:
        logger.warning("EIA fetch failed: %s", ex)

    try:
        import pandas_datareader as pdr

        wcest = pdr.DataReader("WCESTUS1", "fred", start="2015-12-01")
        wcest.columns = ["crude_stocks_mb"]
        wcest["inventory_change"] = wcest["crude_stocks_mb"].diff()
        wcest["inventory_surprise"] = (
            wcest["inventory_change"] - wcest["inventory_change"].rolling(4).mean()
        ) / wcest["inventory_change"].rolling(52).std().replace(0, np.nan)
        wcest.to_csv(cache)
        logger.info("EIA from FRED WCESTUS1 -> %s", cache)
        return wcest
    except Exception as ex:
        logger.warning("FRED EIA fetch failed: %s", ex)
    return None


def fetch_ng_price(macro_dir: Path) -> pd.Series | None:
    """天然气期货日频价格。"""
    macro_dir.mkdir(parents=True, exist_ok=True)
    cache = macro_dir / "ng_daily.csv"
    if cache.is_file():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")
        return s.sort_index()
    try:
        import yfinance as yf

        hist = yf.download("NG=F", start="2015-12-01", progress=False, auto_adjust=True)
        if hist.empty:
            return None
        s = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
        s.name = "ng_price"
        s.to_frame().to_csv(cache)
        logger.info("NG price downloaded -> %s", cache)
        return s
    except Exception as ex:
        logger.warning("NG fetch failed: %s", ex)
    return None


def build_inventory_features(
    m5_index: pd.DatetimeIndex,
    m5: pd.DataFrame,
    macro_dir: Path | None = None,
) -> pd.DataFrame:
    """构建原油库存相关特征，对齐 M5 时间戳。"""
    macro_dir = macro_dir or Path("data/macro")
    out = pd.DataFrame(index=m5_index)

    # hours_to_eia：距下次 EIA 报告小时数（循环编码）
    eia_dates = _eia_wednesdays(m5_index.min(), m5_index.max() + pd.Timedelta(days=14))
    hours_to = np.full(len(m5_index), 168.0)
    for i, t in enumerate(m5_index):
        future = eia_dates[eia_dates > t]
        if len(future):
            hours_to[i] = (future[0] - t).total_seconds() / 3600.0
    out["hours_to_eia"] = hours_to
    out["hours_to_eia_sin"] = np.sin(2 * np.pi * hours_to / 168.0)
    out["hours_to_eia_cos"] = np.cos(2 * np.pi * hours_to / 168.0)

    # 周三前效应
    out["is_pre_eia_day"] = (pd.DatetimeIndex(m5_index).dayofweek == 2).astype(np.float32)

    inv = fetch_eia_inventory(macro_dir)
    out["eia_surprise"] = 0.0
    out["eia_rolling_impact"] = 0.0
    out["inventory_change_norm"] = 0.0

    if inv is not None and not inv.empty:
        if "inventory_surprise" in inv.columns:
            daily_surprise = inv["inventory_surprise"].dropna()
            aligned = daily_surprise.reindex(m5_index.normalize().unique(), method="ffill")
            out["eia_surprise"] = aligned.reindex(m5_index, method="ffill").fillna(0).to_numpy()
            if "inventory_change" in inv.columns:
                chg = inv["inventory_change"].dropna()
                aligned_chg = chg.reindex(m5_index.normalize().unique(), method="ffill")
                std = chg.rolling(52).std().replace(0, np.nan).iloc[-1] or 1.0
                out["inventory_change_norm"] = (aligned_chg.reindex(m5_index, method="ffill").fillna(0) / std).to_numpy()
        elif "oil_ret" in inv.columns:
            # 代理：用原油日收益 rolling 作为 surprise 代理
            ret = inv["oil_ret"].dropna()
            surprise = (ret - ret.rolling(5).mean()) / ret.rolling(20).std().replace(0, np.nan)
            aligned = surprise.reindex(m5_index.normalize().unique(), method="ffill")
            out["eia_surprise"] = aligned.reindex(m5_index, method="ffill").fillna(0).to_numpy()

        # eia_rolling_impact：过去 4 次 EIA 后 1h 价格变化均值
        close = m5["close"]
        impacts: list[float] = []
        for eia_t in eia_dates:
            if eia_t not in m5_index and eia_t < m5_index.max():
                idx = m5_index.searchsorted(eia_t)
                if idx < len(m5_index) - 12:
                    p0 = float(close.iloc[idx])
                    p1 = float(close.iloc[min(idx + 12, len(close) - 1)])
                    if p0 > 0:
                        impacts.append((p1 - p0) / p0)
        avg_impact = float(np.mean(impacts[-4:])) if impacts else 0.0
        out["eia_rolling_impact"] = avg_impact

    # OPEC+ 会议哑变量（主要会议日期，简化标记）
    opec_dates = pd.to_datetime([
        "2016-11-30", "2017-05-25", "2017-11-30", "2018-06-22", "2018-12-07",
        "2019-07-01", "2019-12-05", "2020-04-12", "2020-06-06", "2021-03-04",
        "2021-07-18", "2022-10-05", "2023-06-04", "2023-11-30", "2024-06-02",
        "2024-12-01", "2025-04-03",
    ])
    opec_flag = np.zeros(len(m5_index), dtype=np.float32)
    for d in opec_dates:
        mask = (m5_index >= d) & (m5_index < d + pd.Timedelta(days=3))
        opec_flag[mask] = 1.0
    out["opec_meeting"] = opec_flag

    return out.astype(np.float32)
