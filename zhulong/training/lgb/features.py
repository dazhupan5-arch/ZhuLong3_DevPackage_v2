"""DeepSeek 方案：ATR 通道 + EMA + 多周期 + 序列聚合特征。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_BASE = ["return", "hl_ratio", "close_pos", "volume_change"]
FEATURES_TECH = ["rsi", "macd_line", "macd_signal", "macd_hist", "atr"]
FEATURES_ATR_CHANNEL = [
    "channel_width_3x",
    "channel_position",
    "above_upper_1x",
    "above_upper_2x",
    "above_upper_3x",
    "below_lower_1x",
    "below_lower_2x",
    "below_lower_3x",
    "atr_slope",
]
FEATURES_EMA = [
    "ema_diff",
    "price_ema30",
    "price_ema60",
    "ema30_slope",
    "ema_cross",
]
FEATURES_TIME = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
FEATURES_LAG = [
    "ret_mean_10",
    "ret_std_10",
    "ret_mean_20",
    "ret_std_20",
    "ret_slope_10",
    "ret_slope_20",
]
FEATURES_ASYMM_ATR = ["up_atr_ratio", "down_atr_ratio", "atr_asymmetry"]
FEATURES_LAG_BARS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
    "atr_lag1", "atr_lag2", "atr_lag3",
    "atr_std_10",
]
FEATURES_MTF = [
    "m15_ema_bias",
    "m15_rsi",
    "h1_ema_bias",
    "h1_rsi",
    "h1_macd_sign",
    "h4_ema_bias",
    "m15_h1_trend_align",
    "m5_m15_dir_align",
]
FEATURES_REVERSAL = [
    "rsi_oversold",
    "rsi_overbought",
    "bullish_divergence",
    "bearish_divergence",
    "support_distance",
    "resistance_distance",
    "volume_surge",
    "price_ema50_distance",
    "reversal_candle",
    "bullish_engulfing",
    "bearish_engulfing",
    "atr_expansion",
]

FEATURE_COLUMNS_LGB = (
    FEATURES_BASE
    + FEATURES_TECH
    + FEATURES_ATR_CHANNEL
    + FEATURES_EMA
    + FEATURES_TIME
    + FEATURES_LAG
    + FEATURES_LAG_BARS
    + FEATURES_ASYMM_ATR
    + FEATURES_MTF
)
FEATURE_COLUMNS_LGB_V13 = FEATURE_COLUMNS_LGB + FEATURES_REVERSAL


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _align_to_m5(m5_index: pd.DatetimeIndex, series: pd.Series) -> np.ndarray:
    idx = series.index.searchsorted(m5_index, side="right") - 1
    out = np.zeros(len(m5_index), dtype=np.float32)
    valid = idx >= 0
    out[valid] = series.to_numpy()[idx[valid]]
    return out


def compute_mtf_features(m5: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=m5.index)
    specs = [
        ("15min", "m15", 20),
        ("1h", "h1", 30),
        ("4h", "h4", 30),
    ]
    for rule, prefix, ema_span in specs:
        tf = (
            m5.resample(rule, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        if len(tf) < 10:
            out[f"{prefix}_ema_bias"] = 0.0
            if prefix in ("m15", "h1"):
                out[f"{prefix}_rsi"] = 0.5
            if prefix == "h1":
                out[f"{prefix}_macd_sign"] = 0.0
            continue
        ema = tf["close"].ewm(span=ema_span, adjust=False).mean()
        bias = (tf["close"] - ema) / ema.replace(0, np.nan)
        out[f"{prefix}_ema_bias"] = _align_to_m5(m5.index, bias)
        if prefix in ("m15", "h1"):
            rsi = _rsi(tf["close"], 14) / 100.0
            out[f"{prefix}_rsi"] = _align_to_m5(m5.index, rsi)
        if prefix == "h1":
            ema12 = tf["close"].ewm(span=12, adjust=False).mean()
            ema26 = tf["close"].ewm(span=26, adjust=False).mean()
            sign = np.sign(ema12 - ema26)
            out["h1_macd_sign"] = _align_to_m5(m5.index, sign)
    return out


def add_asymmetric_atr(df: pd.DataFrame, atr_raw: pd.Series) -> pd.DataFrame:
    """上涨/下跌波动不对称特征。"""
    prev_close = df["close"].shift(1)
    up_move = (df["high"] - prev_close).clip(lower=0)
    down_move = (prev_close - df["low"]).clip(lower=0)
    up_atr = up_move.rolling(14).mean() / df["close"].replace(0, np.nan)
    down_atr = down_move.rolling(14).mean() / df["close"].replace(0, np.nan)
    df["up_atr_ratio"] = up_atr
    df["down_atr_ratio"] = down_atr
    denom = (up_atr + down_atr).replace(0, np.nan)
    df["atr_asymmetry"] = (up_atr - down_atr) / denom
    return df


def _rolling_pivot_levels(
    low: pd.Series,
    high: pd.Series,
    close: pd.Series,
    window: int = 50,
) -> tuple[pd.Series, pd.Series]:
    """滚动窗口高低点近似支撑/阻力（向量化，适合大规模训练）。"""
    support = low.rolling(window, min_periods=window).min()
    resistance = high.rolling(window, min_periods=window).max()
    return support, resistance


def _divergence_flags(
    close: pd.Series,
    rsi: pd.Series,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """价格极值与 RSI 极值背离（向量化近似）。"""
    price_low = close <= close.rolling(window, min_periods=window).min()
    price_high = close >= close.rolling(window, min_periods=window).max()
    rsi_low = rsi <= rsi.rolling(window, min_periods=window).min()
    rsi_high = rsi >= rsi.rolling(window, min_periods=window).max()
    bull = (price_low & ~rsi_low).astype(float)
    bear = (price_high & ~rsi_high).astype(float)
    return bull, bear


def add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """v13 反转特征（约 12 维）。"""
    rsi_pct = df["rsi"] * 100.0
    df["rsi_oversold"] = (rsi_pct < 30).astype(float)
    df["rsi_overbought"] = (rsi_pct > 70).astype(float)

    bull_div, bear_div = _divergence_flags(df["close"], rsi_pct, window=20)
    df["bullish_divergence"] = bull_div
    df["bearish_divergence"] = bear_div

    support, resistance = _rolling_pivot_levels(df["low"], df["high"], df["close"], window=50)
    df["support_distance"] = (df["close"] - support) / df["close"].replace(0, np.nan)
    df["resistance_distance"] = (resistance - df["close"]) / df["close"].replace(0, np.nan)

    vol_ma20 = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_surge"] = (df["volume"] / vol_ma20 > 1.5).astype(float)

    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    df["price_ema50_distance"] = (df["close"] - ema50) / ema50.replace(0, np.nan)

    body = (df["close"] - df["open"]).abs()
    lower_shadow = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    df["reversal_candle"] = (
        (lower_shadow > body)
        & (df["close"] > df["open"])
    ).astype(float)

    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    df["bullish_engulfing"] = (
        (df["close"] > df["open"])
        & (prev_c < prev_o)
        & (df["open"] <= prev_c)
        & (df["close"] >= prev_o)
    ).astype(float)
    df["bearish_engulfing"] = (
        (df["close"] < df["open"])
        & (prev_c > prev_o)
        & (df["open"] >= prev_c)
        & (df["close"] <= prev_o)
    ).astype(float)

    atr_raw = _atr(df, 14)
    atr_prev = atr_raw.shift(5).replace(0, np.nan)
    df["atr_expansion"] = (atr_raw / atr_prev > 1.2).astype(float)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["return"]
    for k in range(1, 6):
        df[f"ret_lag{k}"] = ret.shift(k)
    for k in range(1, 4):
        df[f"rsi_lag{k}"] = df["rsi"].shift(k)
        df[f"atr_lag{k}"] = df["atr"].shift(k)
    df["atr_std_10"] = df["atr"].rolling(10).std()
    return df


def compute_features(
    m5: pd.DataFrame,
    include_mtf: bool = True,
    include_reversal: bool = False,
) -> pd.DataFrame:
    df = m5.copy()
    df["return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["hl_ratio"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / hl
    vol_ma = df["volume"].rolling(5).mean().replace(0, np.nan)
    df["volume_change"] = df["volume"] / vol_ma

    df["rsi"] = _rsi(df["close"], 14) / 100.0
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd_line / df["close"]
    df["macd_signal"] = macd_signal / df["close"]
    df["macd_hist"] = (macd_line - macd_signal) / df["close"]

    atr = _atr(df, 14)
    df["atr"] = atr / df["close"]
    df = add_asymmetric_atr(df, atr)
    ema30 = df["close"].ewm(span=30, adjust=False).mean()
    ema60 = df["close"].ewm(span=60, adjust=False).mean()

    for mult, suffix in [(1, "1x"), (2, "2x"), (3, "3x")]:
        upper = ema30 + mult * atr
        lower = ema30 - mult * atr
        df[f"above_upper_{suffix}"] = (df["close"] > upper).astype(float)
        df[f"below_lower_{suffix}"] = (df["close"] < lower).astype(float)

    upper3 = ema30 + 3 * atr
    lower3 = ema30 - 3 * atr
    width3 = (upper3 - lower3).replace(0, np.nan)
    df["channel_width_3x"] = width3 / ema30.replace(0, np.nan)
    df["channel_position"] = (df["close"] - lower3) / width3
    atr_shift = atr.shift(5).replace(0, np.nan)
    df["atr_slope"] = (atr - atr_shift) / atr_shift

    df["ema_diff"] = (ema30 - ema60) / ema60.replace(0, np.nan)
    df["price_ema30"] = (df["close"] - ema30) / ema30.replace(0, np.nan)
    df["price_ema60"] = (df["close"] - ema60) / ema60.replace(0, np.nan)
    ema30_shift = ema30.shift(5)
    df["ema30_slope"] = (ema30 - ema30_shift) / ema30_shift.replace(0, np.nan)
    cross = np.sign(ema30 - ema60) - np.sign(ema30.shift(1) - ema60.shift(1))
    df["ema_cross"] = cross.clip(-1, 1)

    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    ret = df["return"]
    df["ret_mean_10"] = ret.rolling(10).mean()
    df["ret_std_10"] = ret.rolling(10).std()
    df["ret_mean_20"] = ret.rolling(20).mean()
    df["ret_std_20"] = ret.rolling(20).std()
    df["ret_slope_10"] = ret.diff(10) / 10.0
    df["ret_slope_20"] = ret.diff(20) / 20.0
    df = add_lag_features(df)

    if include_mtf:
        mtf = compute_mtf_features(m5)
        df = pd.concat([df, mtf], axis=1)
        m15_sign = np.sign(df["m15_ema_bias"].fillna(0))
        h1_sign = np.sign(df["h1_ema_bias"].fillna(0))
        df["m15_h1_trend_align"] = (m15_sign == h1_sign).astype(float)
        m5_sign = np.sign(df["price_ema30"].fillna(0))
        df["m5_m15_dir_align"] = (m5_sign == m15_sign).astype(float)
    else:
        df["m15_h1_trend_align"] = 0.0
        df["m5_m15_dir_align"] = 0.0
        for col in FEATURES_MTF:
            if col not in df.columns:
                df[col] = 0.0

    if include_reversal:
        df = add_reversal_features(df)
        cols = FEATURE_COLUMNS_LGB_V13
    else:
        cols = FEATURE_COLUMNS_LGB

    feats = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("features computed: %s rows x %s cols", len(feats), len(cols))
    return feats
