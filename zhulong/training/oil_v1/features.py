"""USOIL v1 特征：v8 基础 + 4x ATR 通道 + 波动率自适应 + 库存 + 季节性。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zhulong.training.lgb.features import _atr, _rsi, compute_features, compute_mtf_features
from zhulong.training.v8.features import imf_stat_features, load_macro_features, session_features
from zhulong.training.oil_v1.inventory import build_inventory_features, fetch_ng_price

logger = logging.getLogger(__name__)

OIL_EXTRA_COLS = [
    "above_upper_4x", "below_lower_4x",
    "atr_ratio_20_100", "volatility_regime_low", "volatility_regime_high",
    "month_sin", "month_cos",
    "hours_to_eia_sin", "hours_to_eia_cos", "is_pre_eia_day",
    "eia_surprise", "eia_rolling_impact", "inventory_change_norm",
    "opec_meeting", "macro_ng_ret",
]


def _add_4x_atr_channel(m5: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=m5.index)
    atr = _atr(m5, 14)
    ema30 = m5["close"].ewm(span=30, adjust=False).mean()
    upper4 = ema30 + 4 * atr
    lower4 = ema30 - 4 * atr
    df["above_upper_4x"] = (m5["close"] > upper4).astype(np.float32)
    df["below_lower_4x"] = (m5["close"] < lower4).astype(np.float32)
    return df


def _volatility_adaptive(m5: pd.DataFrame) -> pd.DataFrame:
    atr = _atr(m5, 14) / m5["close"].replace(0, np.nan)
    atr_ma20 = atr.rolling(20).mean()
    atr_ma100 = atr.rolling(100).mean().replace(0, np.nan)
    ratio = (atr_ma20 / atr_ma100).fillna(1.0)
    q33 = ratio.quantile(0.33)
    q67 = ratio.quantile(0.67)
    return pd.DataFrame(
        {
            "atr_ratio_20_100": ratio.astype(np.float32),
            "volatility_regime_low": (ratio < q33).astype(np.float32),
            "volatility_regime_high": (ratio > q67).astype(np.float32),
        },
        index=m5.index,
    )


def _seasonality(index: pd.DatetimeIndex) -> pd.DataFrame:
    month = index.month
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def _load_ng_macro(m5_index: pd.DatetimeIndex, macro_dir: Path) -> pd.Series:
    ng = fetch_ng_price(macro_dir)
    if ng is None or ng.empty:
        return pd.Series(0.0, index=m5_index, name="macro_ng_ret")
    ret = ng.pct_change().dropna()
    aligned = ret.reindex(m5_index.normalize().unique(), method="ffill")
    return aligned.reindex(m5_index, method="ffill").fillna(0).rename("macro_ng_ret")


def build_oil_v1_features(
    m5: pd.DataFrame,
    imf: pd.DataFrame,
    macro_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    macro_dir = macro_dir or Path("data/macro")

    base = compute_features(m5, include_mtf=True)
    imf_stats = imf_stat_features(imf.reindex(m5.index, method="ffill"))
    sessions = session_features(m5.index)
    macro = load_macro_features(m5.index, macro_dir)

    extra_4x = _add_4x_atr_channel(m5)
    vol_adapt = _volatility_adaptive(m5)
    season = _seasonality(m5.index)
    inventory = build_inventory_features(m5.index, m5, macro_dir)
    ng_ret = _load_ng_macro(m5.index, macro_dir).to_frame()

    combined = (
        base.join(imf_stats, how="inner")
        .join(sessions, how="inner")
        .join(macro, how="inner")
        .join(extra_4x, how="inner")
        .join(vol_adapt, how="inner")
        .join(season, how="inner")
        .join(inventory, how="inner")
        .join(ng_ret, how="inner")
    )
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    cols = list(combined.columns)
    logger.info("oil v1 features: %s rows x %s cols", len(combined), len(cols))
    return combined, cols
