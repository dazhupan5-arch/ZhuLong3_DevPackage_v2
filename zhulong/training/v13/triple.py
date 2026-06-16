"""v13 三重屏障训练共享工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.v10.backtest import MIN_ATR_PCT
from zhulong.training.v11.train import proba_to_directions

TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2025-12-31 23:59:59")
THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.75, 0.8]
PRECISION_THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.75, 0.8]
CLASS_WEIGHT_MAP = {0: 1.0, 1: 2.0, 2: 2.0}

V13_TRIPLE_XGB = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "min_child_weight": 3,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
}


def class_sample_weights(y: np.ndarray, use_class_boost: bool = True) -> np.ndarray:
    """频率逆权重 × 多空 2× 加权。"""
    weights = np.ones(len(y), dtype=np.float64)
    for label in (0, 1, 2):
        cnt = max(int((y == label).sum()), 1)
        weights[y == label] = len(y) / (3.0 * cnt)
    if use_class_boost:
        for label, mult in CLASS_WEIGHT_MAP.items():
            weights[y == label] *= mult
    return weights / weights.mean()


def h1_ema50_trend_flags(m5: pd.DataFrame) -> pd.DataFrame:
    """H1 收盘价相对 EMA50 + 斜率方向。"""
    h1 = m5["close"].resample("1h", label="right", closed="right").last().dropna()
    ema50 = h1.ewm(span=50, adjust=False).mean()
    slope = ema50.diff(3)
    flags = pd.DataFrame(
        {
            "long_ok": ((h1 > ema50) & (slope > 0)).astype(np.int8),
            "short_ok": ((h1 < ema50) & (slope < 0)).astype(np.int8),
        },
        index=h1.index,
    )
    return flags.reindex(m5.index, method="ffill").fillna(0)


def apply_trend_filter_v3(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
) -> np.ndarray:
    flags = h1_ema50_trend_flags(m5)
    out = directions.copy()
    for i, t in enumerate(times):
        if out[i] == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            out[i] = 0
            continue
        row = flags.iloc[idx]
        if out[i] == 1 and int(row["long_ok"]) < 1:
            out[i] = 0
        elif out[i] == -1 and int(row["short_ok"]) < 1:
            out[i] = 0
    return out


def postprocess_directions(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    proba: np.ndarray,
    long_thr: float,
    short_thr: float,
    use_trend_filter: bool = True,
) -> np.ndarray:
    dirs = proba_to_directions(proba, long_thr, short_thr)
    dirs = atr_ok(m5, times, dirs)
    if use_trend_filter:
        dirs = apply_trend_filter_v3(m5, times, dirs)
    return dirs


def atr_ok(m5: pd.DataFrame, times: pd.DatetimeIndex, dirs: np.ndarray) -> np.ndarray:
    atr = _atr_series(m5)
    close = m5["close"]
    out = dirs.copy()
    for i, t in enumerate(times):
        if out[i] == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            out[i] = 0
            continue
        a, c = float(atr.iloc[idx]), float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT:
            out[i] = 0
    return out


def _threshold_metrics(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    thr: float,
    days: float,
    use_trend_filter: bool,
) -> dict | None:
    dirs = postprocess_directions(m5, times, proba, thr, thr, use_trend_filter=use_trend_filter)
    pred = np.zeros(len(dirs), dtype=int)
    pred[dirs == 1] = 1
    pred[dirs == -1] = 2
    mask = pred > 0
    n_sig = int(mask.sum())
    if n_sig == 0:
        return None
    spd = n_sig / days
    long_m, short_m = dirs == 1, dirs == -1
    lp = float((y_true[long_m] == 1).mean()) if long_m.any() else 0.0
    sp = float((y_true[short_m] == 2).mean()) if short_m.any() else 0.0
    wprec = (lp * long_m.sum() + sp * short_m.sum()) / max(n_sig, 1)
    trade_recall = float(
        recall_score(y_true[mask], pred[mask], labels=[1, 2], average="macro", zero_division=0)
    )
    trade_f1 = float(
        f1_score(y_true[mask], pred[mask], labels=[1, 2], average="macro", zero_division=0)
    )
    return {
        "threshold": thr,
        "weighted_precision": wprec,
        "long_precision": lp,
        "short_precision": sp,
        "trade_f1": trade_f1,
        "trade_recall": trade_recall,
        "n_signals": n_sig,
        "signals_per_day": spd,
    }


def _asymmetric_metrics(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    long_thr: float,
    short_thr: float,
    days: float,
    use_trend_filter: bool,
) -> dict | None:
    dirs = postprocess_directions(
        m5, times, proba, long_thr, short_thr, use_trend_filter=use_trend_filter
    )
    pred = np.zeros(len(dirs), dtype=int)
    pred[dirs == 1] = 1
    pred[dirs == -1] = 2
    mask = pred > 0
    n_sig = int(mask.sum())
    if n_sig == 0:
        return None
    spd = n_sig / days
    long_m, short_m = dirs == 1, dirs == -1
    n_long, n_short = int(long_m.sum()), int(short_m.sum())
    lp = float((y_true[long_m] == 1).mean()) if n_long else 0.0
    sp = float((y_true[short_m] == 2).mean()) if n_short else 0.0
    wprec = (lp * n_long + sp * n_short) / max(n_sig, 1)
    trade_recall = float(
        recall_score(y_true[mask], pred[mask], labels=[1, 2], average="macro", zero_division=0)
    )
    return {
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "threshold": long_thr,
        "weighted_precision": wprec,
        "long_precision": lp,
        "short_precision": sp,
        "trade_recall": trade_recall,
        "n_signals": n_sig,
        "n_long_signals": n_long,
        "n_short_signals": n_short,
        "signals_per_day": spd,
        "min_side_precision": min(lp if n_long else 0.0, sp if n_short else 0.0),
    }


def tune_precision_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    thresholds: list[float] | None = None,
    target_precision: float = 0.55,
    max_signals_per_day: float = 6.0,
    use_trend_filter: bool = True,
) -> tuple[float, float, list[dict], dict]:
    """精确率优先：多空独立阈值，均达标时取最高平均阈值。"""
    grid = thresholds or PRECISION_THRESHOLD_GRID
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    rows: list[dict] = []
    qualified: list[tuple[float, float, dict]] = []

    for long_thr in grid:
        for short_thr in grid:
            row = _asymmetric_metrics(
                proba, y_true, times, m5, long_thr, short_thr, days, use_trend_filter
            )
            if row is None:
                continue
            rows.append(row)
            if (
                row["n_long_signals"] >= 5
                and row["n_short_signals"] >= 5
                and row["long_precision"] >= target_precision
                and row["short_precision"] >= target_precision
                and row["signals_per_day"] <= max_signals_per_day
            ):
                qualified.append((long_thr, short_thr, row))

    if qualified:
        _, _, best_metrics = max(qualified, key=lambda x: (x[0] + x[1], x[2]["weighted_precision"]))
        return best_metrics["long_threshold"], best_metrics["short_threshold"], rows, best_metrics

    eligible = [
        r for r in rows
        if r["signals_per_day"] <= max_signals_per_day
        and r["n_long_signals"] >= 3
        and r["n_short_signals"] >= 3
    ]
    if eligible:
        best_metrics = max(
            eligible,
            key=lambda x: (x["min_side_precision"], x["weighted_precision"], -(x["signals_per_day"])),
        )
        return best_metrics["long_threshold"], best_metrics["short_threshold"], rows, best_metrics

    if rows:
        best_metrics = max(rows, key=lambda x: (x["min_side_precision"], x["weighted_precision"]))
        return best_metrics["long_threshold"], best_metrics["short_threshold"], rows, best_metrics

    return 0.6, 0.6, rows, {"long_threshold": 0.6, "short_threshold": 0.6, "weighted_precision": 0.0}


def tune_f1_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    thresholds: list[float] | None = None,
    max_signals_per_day: float = 8.0,
) -> tuple[float, float, list[dict], dict]:
    return tune_precision_thresholds(
        proba, y_true, times, m5, thresholds, target_precision=0.0, max_signals_per_day=max_signals_per_day
    )
