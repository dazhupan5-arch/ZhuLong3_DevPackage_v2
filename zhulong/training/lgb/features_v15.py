"""V15 特征：V13 68 维 + 8 维日内/Regime 扩展。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, _atr, _rsi, compute_features

FEATURES_V15_EXT = [
    "ret_12bar",
    "ret_48bar",
    "ret_day",
    "dist_sma200_atr",
    "adx_14",
    "vol_ratio",
    "lower_highs_12",
    "range_expansion",
]
FEATURE_COLUMNS_V15 = FEATURE_COLUMNS_LGB_V13 + FEATURES_V15_EXT


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().replace(0, np.nan)
    plus = pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus = pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = (plus - minus).abs() / (plus + minus).replace(0, np.nan) * 100
    return dx.rolling(period).mean().fillna(0) / 100.0


def add_v15_extension_features(m5: pd.DataFrame) -> pd.DataFrame:
    close = m5["close"]
    atr_raw = _atr(m5, 14)
    atr_pct = atr_raw / close.replace(0, np.nan)
    ext = pd.DataFrame(index=m5.index)
    ext["ret_12bar"] = close.pct_change(12)
    ext["ret_48bar"] = close.pct_change(48)
    day_open = close.groupby(m5.index.normalize()).transform("first")
    ext["ret_day"] = (close - day_open) / day_open.replace(0, np.nan)
    sma200 = close.rolling(200, min_periods=50).mean()
    ext["dist_sma200_atr"] = (close - sma200) / atr_raw.replace(0, np.nan)
    ext["adx_14"] = _adx(m5)
    ext["vol_ratio"] = atr_pct / atr_pct.rolling(20).mean().replace(0, np.nan)
    roll_high = m5["high"]
    lower_highs = (roll_high < roll_high.shift(1)).rolling(12).sum() / 12.0
    ext["lower_highs_12"] = lower_highs
    day_high = m5["high"].groupby(m5.index.normalize()).transform("max")
    day_low = m5["low"].groupby(m5.index.normalize()).transform("min")
    day_range = (day_high - day_low) / close.replace(0, np.nan)
    avg_range = day_range.rolling(5 * 288, min_periods=288).mean()
    ext["range_expansion"] = day_range / avg_range.replace(0, np.nan)
    return ext.replace([np.inf, -np.inf], np.nan)


def compute_features_v15(m5: pd.DataFrame) -> pd.DataFrame:
    base = compute_features(m5, include_mtf=True, include_reversal=True)
    ext = add_v15_extension_features(m5).reindex(base.index)
    out = pd.concat([base, ext[FEATURES_V15_EXT]], axis=1)
    out = out[FEATURE_COLUMNS_V15].replace([np.inf, -np.inf], np.nan).dropna()
    return out
