"""V17 训练标签：方向回归 + 位置二分类。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.agent.structure_analyzer import FEATURE_NAMES


def make_direction_regression_labels(
    close: np.ndarray,
    atr: np.ndarray,
    *,
    horizon_bars: int = 12,
    tanh_scale: float = 3.0,
) -> np.ndarray:
    """future_return/ATR 经 tanh 压缩到 [-1, +1]。"""
    n = len(close)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n - horizon_bars):
        c0 = float(close[i])
        c1 = float(close[i + horizon_bars])
        if c0 <= 0:
            continue
        future_return = (c1 - c0) / c0
        atr_norm = max(float(atr[i]), c0 * 0.0001)
        raw_score = future_return / atr_norm * 100.0
        y[i] = np.tanh(raw_score / max(tanh_scale, 1e-9))
    return y


def make_location_binary_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    direction_series: np.ndarray,
    *,
    tp_atr: float = 2.0,
    sl_atr: float = 1.2,
    max_bars: int = 48,
) -> np.ndarray:
    """给定方向，先触 TP=1，先触 SL=0。"""
    n = len(close)
    y = np.zeros(n, dtype=np.int8)
    dirs = np.asarray(direction_series, dtype=np.int8)

    for i in range(n - max_bars):
        d = int(dirs[i])
        if d == 0:
            continue
        entry = float(close[i])
        a = max(float(atr[i]), entry * 0.0001)
        if d > 0:
            tp_price = entry + tp_atr * a
            sl_price = entry - sl_atr * a
        else:
            tp_price = entry - tp_atr * a
            sl_price = entry + sl_atr * a

        label = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            h, l = float(high[j]), float(low[j])
            if d > 0:
                if h >= tp_price:
                    label = 1
                    break
                if l <= sl_price:
                    label = 0
                    break
            else:
                if l <= tp_price:
                    label = 1
                    break
                if h >= sl_price:
                    label = 0
                    break
        y[i] = label
    return y


def direction_series_from_scores(
    direction_score: np.ndarray,
    threshold: float = 0.35,
) -> np.ndarray:
    scores = np.asarray(direction_score, dtype=np.float32)
    out = np.zeros(len(scores), dtype=np.int8)
    out[scores >= threshold] = 1
    out[scores <= -threshold] = -1
    return out


def atr_percentile_series(atr: np.ndarray, window: int = 2880) -> np.ndarray:
    """滚动 ATR 分位数（默认约 10 日 M5）。"""
    s = pd.Series(np.asarray(atr, dtype=np.float64))
    pct = s.rolling(window, min_periods=max(window // 10, 50)).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) if len(x) else 0.5,
        raw=False,
    )
    return pct.fillna(0.5).to_numpy(dtype=np.float32)


def build_location_feature_matrix(
    struct: np.ndarray,
    pos_in_range: np.ndarray,
    direction_score: np.ndarray,
    atr_percentile: np.ndarray,
) -> np.ndarray:
    """批量构建 LocationGate 特征 (N, 15)。"""
    n = len(struct)
    out = np.zeros((n, 15), dtype=np.float32)
    for i in range(n):
        vec = np.asarray(struct[i], dtype=np.float32).reshape(-1)
        score = float(direction_score[i])
        direction = 0.0 if abs(score) < 1e-9 else (1.0 if score > 0 else -1.0)
        out[i] = [
            float(pos_in_range[i]),
            float(vec[3]) if vec.size > 3 else 0.0,
            float(vec[4]) if vec.size > 4 else 0.0,
            float(vec[5]) if vec.size > 5 else 0.3,
            float(vec[6]) if vec.size > 6 else 0.3,
            float(vec[26]) if vec.size > 26 else 0.0,
            float(atr_percentile[i]),
            float(vec[17]) if vec.size > 17 else 0.0,
            float(vec[18]) if vec.size > 18 else 0.0,
            float(vec[19]) if vec.size > 19 else 0.0,
            float(vec[20]) if vec.size > 20 else 0.0,
            direction,
            abs(score),
            float(vec[16]) if vec.size > 16 else 1.0,
            float(vec[1]) if vec.size > 1 else 0.0,
        ]
    return out


def summarize_labels(
    direction_score: np.ndarray,
    location_labels: np.ndarray,
    direction_series: np.ndarray | None = None,
) -> dict:
    dirs = direction_series if direction_series is not None else direction_series_from_scores(direction_score)
    mask = dirs != 0
    trade_rate = float(location_labels[mask].mean()) if mask.any() else 0.0
    return {
        "direction_score_mean": float(np.mean(direction_score)),
        "direction_score_std": float(np.std(direction_score)),
        "direction_nonzero_rate": float(mask.mean()),
        "location_label_positive_rate": trade_rate,
        "feature_names": list(FEATURE_NAMES),
    }
