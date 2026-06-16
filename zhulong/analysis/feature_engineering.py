"""V13 增强特征：差分、高阶统计、交互、宏观、形态。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_DIFF = ["ret_1", "ret_2", "ret_3"]
FEATURES_ROLLING_STATS = [
    "close_mean_10",
    "close_std_10",
    "close_skew_10",
    "close_kurt_10",
    "close_mean_20",
    "close_std_20",
]
FEATURES_INTERACTION = [
    "dxy_x_volume",
    "rsi_x_hl_width",
    "macd_x_atr",
    "ema_bias_x_volume",
]
FEATURES_MACRO = ["dxy_return", "us10y_chg", "vix_chg"]
FEATURES_PATTERN_EXTRA = ["has_double_top", "has_double_bottom", "adx_norm"]
FEATURES_KEY_LEVEL = [
    "dist_to_support",
    "dist_to_resistance",
    "near_support",
    "near_resistance",
    "price_position",
]

FEATURES_ENHANCED = (
    FEATURES_DIFF
    + FEATURES_ROLLING_STATS
    + FEATURES_INTERACTION
    + FEATURES_MACRO
    + FEATURES_PATTERN_EXTRA
)
FEATURES_ENHANCED_WITH_LEVELS = FEATURES_ENHANCED + FEATURES_KEY_LEVEL


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()


def _local_extrema_flags(series: pd.Series, window: int = 20, tol: float = 0.002) -> tuple[pd.Series, pd.Series]:
    """双顶/双底代理：前后半窗极值相近且现价贴近极值区。"""
    half = max(window // 2, 5)
    roll_max = series.rolling(window, min_periods=window).max()
    roll_min = series.rolling(window, min_periods=window).min()
    prev_max = series.shift(half).rolling(half, min_periods=half).max()
    prev_min = series.shift(half).rolling(half, min_periods=half).min()
    denom_hi = roll_max.replace(0, np.nan)
    denom_lo = roll_min.replace(0, np.nan)
    near_two_peaks = (roll_max - prev_max).abs() / denom_hi < tol
    near_two_troughs = (roll_min - prev_min).abs() / denom_lo.abs() < tol
    double_top = (near_two_peaks & (series >= roll_max * (1 - tol))).astype(np.float32)
    double_bottom = (near_two_troughs & (series <= roll_min * (1 + tol))).astype(np.float32)
    return double_top, double_bottom


def load_macro_aligned(m5_index: pd.DatetimeIndex, root: Path | None = None) -> pd.DataFrame:
    """加载 DXY / US10Y / VIX 并对齐到 M5（前向填充）。"""
    root = root or Path(__file__).resolve().parents[2]
    macro_dir = root / "data" / "macro"
    out = pd.DataFrame(index=m5_index)
    specs = [
        ("dxy_return", ["dxy_m5.csv", "DXY_M5.csv"], "close"),
        ("us10y_chg", ["us10y_m5.csv", "US10Y_M5.csv"], "close"),
        ("vix_chg", ["vix_m5.csv", "VIX_M5.csv"], "close"),
    ]
    for col, files, price_col in specs:
        loaded = False
        for fname in files:
            p = macro_dir / fname
            if not p.is_file():
                continue
            try:
                raw = pd.read_csv(p, index_col=0, parse_dates=True)
                if price_col not in raw.columns and "close" in raw.columns:
                    price_col = "close"
                if price_col not in raw.columns:
                    continue
                s = raw[price_col].astype(float)
                s = s.reindex(m5_index, method="ffill")
                out[col] = s.pct_change().fillna(0.0)
                loaded = True
                break
            except Exception as ex:
                logger.warning("macro load failed %s: %s", p, ex)
        if not loaded:
            out[col] = 0.0
    return out


def add_key_level_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """支撑/阻力距离、区间位置（ATR 归一化）。"""
    high, low, close = df["high"], df["low"], df["close"]
    atr = df.get("atr")
    if atr is None:
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    atr_safe = atr.replace(0, np.nan)

    roll_hi = high.rolling(lookback, min_periods=20).max()
    roll_lo = low.rolling(lookback, min_periods=20).min()
    pivot_hi = high.where(
        (high >= high.shift(1)) & (high >= high.shift(-1))
        & (high >= high.shift(2)) & (high >= high.shift(-2))
    )
    pivot_lo = low.where(
        (low <= low.shift(1)) & (low <= low.shift(-1))
        & (low <= low.shift(2)) & (low <= low.shift(-2))
    )
    resistance = pivot_hi.ffill().combine_first(roll_hi)
    support = pivot_lo.ffill().combine_first(roll_lo)

    df["dist_to_support"] = ((close - support) / atr_safe).fillna(0.0)
    df["dist_to_resistance"] = ((resistance - close) / atr_safe).fillna(0.0)
    df["near_support"] = (df["dist_to_support"] < 0.5).astype(np.float32)
    df["near_resistance"] = (df["dist_to_resistance"] < 0.5).astype(np.float32)
    span = (roll_hi - roll_lo).replace(0, np.nan)
    df["price_position"] = ((close - roll_lo) / span).fillna(0.5)
    return df


def add_enhanced_features(
    df: pd.DataFrame,
    root: Path | None = None,
    *,
    include_key_levels: bool = False,
) -> pd.DataFrame:
    """在已有基础列上叠加增强特征（需含 close/rsi/atr/volume 等）。"""
    close = df["close"]
    ret_1 = close.pct_change()
    df["ret_1"] = ret_1
    df["ret_2"] = ret_1.diff()
    df["ret_3"] = df["ret_2"].diff()

    for w, prefix in [(10, "10"), (20, "20")]:
        roll = close.rolling(w, min_periods=w)
        df[f"close_mean_{prefix}"] = roll.mean() / close.replace(0, np.nan)
        df[f"close_std_{prefix}"] = roll.std() / close.replace(0, np.nan)
    roll10 = close.rolling(10, min_periods=10)
    df["close_skew_10"] = roll10.skew()
    df["close_kurt_10"] = roll10.kurt()

    macro = load_macro_aligned(df.index, root)
    vol_ratio = df.get("volume_change", df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan))
    hl_width = (df["high"] - df["low"]) / close.replace(0, np.nan)
    rsi = df.get("rsi", pd.Series(0.5, index=df.index))
    df["dxy_x_volume"] = macro["dxy_return"] * vol_ratio
    df["rsi_x_hl_width"] = rsi * hl_width
    df["macd_x_atr"] = df.get("macd_hist", 0.0) * df.get("atr", 0.0)
    df["ema_bias_x_volume"] = df.get("price_ema30", 0.0) * vol_ratio
    df["dxy_return"] = macro["dxy_return"]
    df["us10y_chg"] = macro["us10y_chg"]
    df["vix_chg"] = macro["vix_chg"]

    dtop, dbot = _local_extrema_flags(close, window=20)
    df["has_double_top"] = dtop
    df["has_double_bottom"] = dbot
    adx = _adx(df, 14)
    df["adx_norm"] = adx / 100.0

    if include_key_levels:
        df = add_key_level_features(df)
    return df
