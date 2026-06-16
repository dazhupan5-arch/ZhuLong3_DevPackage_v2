"""v8 特征：LGB 51 维 + IMF 统计 + 时段 + 宏观（可选）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB, compute_features

logger = logging.getLogger(__name__)

IMF_STAT_WINDOW = 20
SESSION_COLS = ["session_asia", "session_london", "session_ny"]
MACRO_COLS = ["macro_vix", "macro_dxy", "macro_us10y", "macro_gold_ret"]


def imf_stat_features(imf: pd.DataFrame, window: int = IMF_STAT_WINDOW) -> pd.DataFrame:
    out = pd.DataFrame(index=imf.index)
    for col in imf.columns:
        s = imf[col]
        out[f"{col}_mean"] = s.rolling(window).mean()
        out[f"{col}_std"] = s.rolling(window).std()
        out[f"{col}_skew"] = s.rolling(window).skew()
        out[f"{col}_kurt"] = s.rolling(window).kurt()
        out[f"{col}_energy"] = (s**2).rolling(window).sum()
        out[f"{col}_roc"] = s.pct_change(window)
    return out


def session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour
    return pd.DataFrame(
        {
            "session_asia": ((hour >= 0) & (hour < 8)).astype(np.float32),
            "session_london": ((hour >= 8) & (hour < 16)).astype(np.float32),
            "session_ny": ((hour >= 13) & (hour < 21)).astype(np.float32),
        },
        index=index,
    )


def load_macro_features(m5_index: pd.DatetimeIndex, macro_dir: Path | None = None) -> pd.DataFrame:
    """宏观因子：优先读本地 CSV；实机默认不拉 yfinance（避免 60s+ 阻塞信号调度）。"""
    out = pd.DataFrame(0.0, index=m5_index, columns=MACRO_COLS)
    macro_dir = macro_dir or Path("data/macro")
    local = macro_dir / "macro_daily.csv"
    if local.is_file():
        daily = pd.read_csv(local, index_col=0, parse_dates=True).sort_index()
        aligned = daily.reindex(m5_index.normalize().unique(), method="ffill")
        for col in MACRO_COLS:
            src = col.replace("macro_", "")
            if src in aligned.columns:
                m = aligned[src].reindex(m5_index, method="ffill")
                out[col] = m.to_numpy(dtype=np.float32)
        logger.info("macro from %s", local)
        return out

    if os.environ.get("ZHULONG_ALLOW_YFINANCE", "").lower() not in ("1", "true", "yes"):
        logger.warning("macro unavailable, using zeros (set ZHULONG_ALLOW_YFINANCE=1 to enable yfinance): %s", local)
        return out

    try:
        import yfinance as yf

        tickers = {"macro_vix": "^VIX", "macro_dxy": "DX-Y.NYB", "macro_us10y": "^TNX"}
        daily_ix = pd.date_range(m5_index.min().normalize(), m5_index.max().normalize(), freq="D")
        macro_daily = pd.DataFrame(index=daily_ix)
        for col, tic in tickers.items():
            hist = yf.download(tic, start=daily_ix.min(), end=daily_ix.max() + pd.Timedelta(days=1), progress=False)
            if not hist.empty:
                s = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
                macro_daily[col] = s.reindex(daily_ix, method="ffill")
        if "macro_gold_ret" not in macro_daily.columns:
            macro_daily["macro_gold_ret"] = 0.0
        macro_dir.mkdir(parents=True, exist_ok=True)
        macro_daily.to_csv(local)
        for col in MACRO_COLS:
            if col in macro_daily.columns:
                out[col] = macro_daily[col].reindex(m5_index, method="ffill").to_numpy(dtype=np.float32)
        logger.info("macro downloaded via yfinance -> %s", local)
    except Exception as ex:
        logger.warning("macro unavailable, using zeros: %s", ex)
    return out


def build_v8_features(
    m5: pd.DataFrame,
    imf: pd.DataFrame,
    macro_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    base = compute_features(m5, include_mtf=True)
    imf_stats = imf_stat_features(imf.reindex(m5.index, method="ffill"))
    sessions = session_features(m5.index)
    macro = load_macro_features(m5.index, macro_dir)

    combined = base.join(imf_stats, how="inner").join(sessions, how="inner").join(macro, how="inner")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    cols = list(combined.columns)
    logger.info("v8 features: %s rows x %s cols", len(combined), len(cols))
    return combined, cols
