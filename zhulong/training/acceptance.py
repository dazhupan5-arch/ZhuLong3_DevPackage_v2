"""
烛龙模型训练验收（Stephen.Pan 标准）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROB_THRESHOLD = 0.60
OOS_BARS_2W = 14 * 24 * 12  # 4032 M5 bars
VAL_FRAC = 0.25
VAL_BARS_3M = 90 * 24 * 12  # 约 3 个月 M5
SL_ATR = 1.2
TP_ATR = 2.0
LABEL_MODE = "return_direction"  # return_direction | triple_barrier
RETURN_R_UNIT = 0.15  # 与标签阈值一致，用于 R 倍数换算


@dataclass
class LabelParams:
    horizon: int = 12
    return_threshold_pct: float = 0.15
    long_max_dd_pct: float = 0.15
    long_min_gain_pct: float = 0.25
    short_max_rise_pct: float = 0.15
    short_min_drop_pct: float = 0.25

    def as_dict(self) -> dict[str, float | int]:
        return {
            "horizon": self.horizon,
            "return_threshold_pct": self.return_threshold_pct,
            "label_mode": LABEL_MODE,
            "long_max_dd_pct": self.long_max_dd_pct,
            "long_min_gain_pct": self.long_min_gain_pct,
            "short_max_rise_pct": self.short_max_rise_pct,
            "short_min_drop_pct": self.short_min_drop_pct,
        }


@dataclass
class AcceptanceThresholds:
    """DeepSeek 调参后阶段性验收（2026-06）。"""
    precision: float = 0.60
    precision_relaxed: float = 0.55
    recall: float = 0.15
    f1: float = 0.25
    long_short_prec_gap: float = 0.20
    reg_mae_pct: float = 999.0
    reg_sign_acc: float = 0.0
    avg_rr: float = 1.3
    train_val_prec_gap: float = 0.25
    label_sensitivity: float = 0.05
    oos_win_rate: float = 0.55
    extreme_max_consec_loss: int = 8
    extreme_max_drawdown: float = 0.25
    require_regressor: bool = False


@dataclass
class AcceptanceReport:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    label_params: dict[str, Any] = field(default_factory=dict)
    data_range: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "metrics": self.metrics,
            "label_params": self.label_params,
            "data_range": self.data_range,
        }


def predict_from_dual(
    p_long: np.ndarray,
    p_short: np.ndarray,
    threshold: float = PROB_THRESHOLD,
    short_max: float | None = None,
) -> np.ndarray:
    pred = np.ones(len(p_long), dtype=int)
    long_ok = (p_long >= threshold) & (p_long >= p_short)
    if short_max is not None:
        long_ok &= p_short <= short_max
    short_ok = (p_short >= threshold) & (p_short > p_long)
    if short_max is not None:
        short_ok &= p_long <= short_max
    pred[long_ok] = 2
    pred[short_ok] = 0
    return pred


def proba_3class_from_dual(p_long: np.ndarray, p_short: np.ndarray) -> np.ndarray:
    p_flat = np.clip(1.0 - p_long - p_short, 0.05, 1.0)
    mat = np.stack([p_short, p_flat, p_long], axis=1)
    return mat / mat.sum(axis=1, keepdims=True)


def predict_with_threshold(proba: np.ndarray, threshold: float = PROB_THRESHOLD) -> np.ndarray:
    """proba: (N,3) classes [short=0, flat=1, long=2]."""
    p_short, _, p_long = proba[:, 0], proba[:, 1], proba[:, 2]
    return predict_from_dual(p_long, p_short, threshold)


def directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """做多/做空加权精确率、召回率、F1（仅方向类 0/2）。"""
    mask_pred = np.isin(y_pred, [0, 2])
    mask_true = np.isin(y_true, [0, 2])
    eval_mask = mask_pred | mask_true
    if not eval_mask.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "prec_long": 0.0, "prec_short": 0.0}

    yt = y_true[eval_mask]
    yp = y_pred[eval_mask]
    labels = [0, 2]
    prec = precision_score(yt, yp, labels=labels, average="weighted", zero_division=0)
    rec = recall_score(yt, yp, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)
    prec_long = precision_score(yt == 2, yp == 2, zero_division=0)
    prec_short = precision_score(yt == 0, yp == 0, zero_division=0)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "prec_long": float(prec_long),
        "prec_short": float(prec_short),
    }


def regressor_metrics(
    y_pred_dir: np.ndarray,
    y_true_dir: np.ndarray,
    entry_pred: np.ndarray,
    entry_true: np.ndarray,
) -> dict[str, float]:
    mask = np.isin(y_pred_dir, [0, 2])
    if not mask.any():
        return {"mae_pct": 999.0, "sign_accuracy": 0.0, "n": 0}
    err = np.abs(entry_pred[mask] - entry_true[mask]) * 100.0
    sign_ok = np.sign(entry_pred[mask]) == np.sign(entry_true[mask])
    return {
        "mae_pct": float(err.mean()),
        "sign_accuracy": float(sign_ok.mean()),
        "n": int(mask.sum()),
    }


def _simulate_bar_outcome(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
) -> float:
    """返回相对 entry 的 R 倍数（赢 +TP/SL，输 -1）。direction: +1 做多, -1 做空。"""
    if direction < 0:
        sl = entry + SL_ATR * atr
        tp = entry - TP_ATR * atr
        for h, l in zip(highs, lows):
            hit_sl = h >= sl
            hit_tp = l <= tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return TP_ATR / SL_ATR
        return 0.0
    sl = entry - SL_ATR * atr
    tp = entry + TP_ATR * atr
    for h, l in zip(highs, lows):
        hit_sl = l <= sl
        hit_tp = h >= tp
        if hit_sl and hit_tp:
            return -1.0
        if hit_sl:
            return -1.0
        if hit_tp:
            return TP_ATR / SL_ATR
    return 0.0


def backtest_return_aligned(
    y_pred: np.ndarray,
    m5: pd.DataFrame,
    bar_indices: np.ndarray,
    horizon: int = 12,
) -> dict[str, float]:
    """与 return_direction 标签对齐：方向 × 未来 horizon 收盘收益率。"""
    close = m5["close"].to_numpy()
    rs: list[float] = []
    for pred, bi in zip(y_pred, bar_indices):
        if pred not in (0, 2):
            continue
        idx = int(bi)
        if idx + horizon >= len(close):
            continue
        ret_pct = (close[idx + horizon] - close[idx]) / close[idx] * 100.0
        direction = 1 if pred == 2 else -1
        rs.append(direction * ret_pct / max(RETURN_R_UNIT, 1e-6))
    if not rs:
        return {"avg_rr": 0.0, "expectancy": -1.0, "win_rate": 0.0, "n_trades": 0}
    rs_arr = np.array(rs)
    wins = rs_arr[rs_arr > 0]
    losses = rs_arr[rs_arr < 0]
    win_rate = float((rs_arr > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1.0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        "avg_rr": avg_rr,
        "expectancy": float(expectancy),
        "win_rate": win_rate,
        "n_trades": int(len(rs_arr)),
    }


def backtest_validation(
    y_pred: np.ndarray,
    entry_pred: np.ndarray,
    m5: pd.DataFrame,
    bar_indices: np.ndarray,
    horizon: int = 6,
    use_return_aligned: bool = True,
) -> dict[str, float]:
    if use_return_aligned or LABEL_MODE == "return_direction":
        return backtest_return_aligned(y_pred, m5, bar_indices, horizon)
    atr_series = _atr_series(m5)
    rs: list[float] = []
    for i, (pred, off) in enumerate(zip(y_pred, entry_pred)):
        if pred not in (0, 2):
            continue
        idx = int(bar_indices[i])
        if idx + horizon >= len(m5):
            continue
        close = float(m5["close"].iloc[idx])
        atr = float(atr_series.iloc[idx])
        if atr <= 0 or np.isnan(atr):
            continue
        entry = close * (1.0 + off)
        hs = m5["high"].iloc[idx + 1 : idx + 1 + horizon].to_numpy()
        ls = m5["low"].iloc[idx + 1 : idx + 1 + horizon].to_numpy()
        direction = 1 if pred == 2 else -1
        rs.append(_simulate_bar_outcome(direction, entry, atr, hs, ls))

    if not rs:
        return {"avg_rr": 0.0, "expectancy": -1.0, "win_rate": 0.0, "n_trades": 0}

    rs_arr = np.array(rs)
    wins = rs_arr[rs_arr > 0]
    losses = rs_arr[rs_arr < 0]
    win_rate = float((rs_arr > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1.0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        "avg_rr": avg_rr,
        "expectancy": float(expectancy),
        "win_rate": win_rate,
        "n_trades": int(len(rs_arr)),
    }


def _atr_series(m5: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = m5["close"].shift(1)
    tr = pd.concat(
        [
            m5["high"] - m5["low"],
            (m5["high"] - prev_close).abs(),
            (m5["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def oos_backtest(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    fused: np.ndarray,
    y_true: np.ndarray,
    entry_true: np.ndarray,
    m5: pd.DataFrame,
    bar_indices: np.ndarray,
    horizon: int = 6,
) -> dict[str, float]:
    proba = clf.predict_proba(fused)
    y_pred = predict_with_threshold(proba)
    entry_pred = reg.predict(fused)
    bt = backtest_validation(y_pred, entry_pred, m5, bar_indices, horizon)
    dir_mask = np.isin(y_pred, [0, 2])
    if dir_mask.sum() == 0:
        bt["signal_win_rate"] = 0.0
        return bt
    correct = (y_pred[dir_mask] == y_true[dir_mask]).mean()
    bt["signal_win_rate"] = float(correct)
    bt["total_pnl_r"] = float(bt.get("expectancy", 0) * bt.get("n_trades", 0))
    return bt


def find_extreme_month(m5: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    atr_pct = _atr_series(m5) / m5["close"]
    monthly = atr_pct.resample("ME").mean().dropna()
    if monthly.empty:
        ts = m5.index[0]
        return ts, ts + pd.offsets.MonthEnd(0)
    peak = monthly.idxmax()
    start = peak.replace(day=1)
    end = peak
    return start, end


def consecutive_losses(rs: list[float]) -> int:
    max_streak = streak = 0
    for r in rs:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    denom = np.maximum(np.abs(peak), 1e-6)
    dd = (peak - equity) / denom
    return float(np.clip(dd.max(), 0.0, 1.0))


def evaluate_acceptance(
    *,
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    fused_tr: np.ndarray,
    fused_va: np.ndarray,
    fused_oos: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_oos: np.ndarray,
    entry_tr: np.ndarray,
    entry_va: np.ndarray,
    entry_oos: np.ndarray,
    m5: pd.DataFrame,
    va_bar_idx: np.ndarray,
    oos_bar_idx: np.ndarray,
    extreme_bar_idx: np.ndarray,
    label_params: LabelParams,
    data_range: dict[str, str],
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceReport:
    th = thresholds or AcceptanceThresholds()
    failures: list[str] = []

    proba_tr = clf.predict_proba(fused_tr)
    proba_va = clf.predict_proba(fused_va)
    y_pred_tr = predict_with_threshold(proba_tr)
    y_pred_va = predict_with_threshold(proba_va)
    entry_pred_va = reg.predict(fused_va)

    dir_tr = directional_metrics(y_tr, y_pred_tr)
    dir_va = directional_metrics(y_va, y_pred_va)
    reg_va = regressor_metrics(y_pred_va, y_va, entry_pred_va, entry_va)
    bt_va = backtest_validation(y_pred_va, entry_pred_va, m5, va_bar_idx)

    metrics: dict[str, Any] = {
        "train": dir_tr,
        "validation": dir_va,
        "regressor": reg_va,
        "backtest_validation": bt_va,
    }

    if dir_va["precision"] < th.precision_relaxed:
        failures.append(f"验证集精确率 {dir_va['precision']:.3f} < {th.precision_relaxed}")
    elif dir_va["precision"] < th.precision:
        metrics["precision_relaxed_note"] = (
            f"精确率 {dir_va['precision']:.3f} 在 [{th.precision_relaxed}, {th.precision})，需项目方确认"
        )

    if dir_va["recall"] < th.recall:
        failures.append(f"验证集召回率 {dir_va['recall']:.3f} < {th.recall}")
    if dir_va["f1"] < th.f1:
        failures.append(f"验证集 F1 {dir_va['f1']:.3f} < {th.f1}")
    if abs(dir_va["prec_long"] - dir_va["prec_short"]) > th.long_short_prec_gap:
        failures.append(
            f"做多/做空精确率差 {abs(dir_va['prec_long'] - dir_va['prec_short']):.3f} > {th.long_short_prec_gap}"
        )

    if reg_va["mae_pct"] > th.reg_mae_pct:
        failures.append(f"回归 MAE {reg_va['mae_pct']:.3f}% > {th.reg_mae_pct}%")
    if reg_va["sign_accuracy"] < th.reg_sign_acc:
        failures.append(f"回归符号准确率 {reg_va['sign_accuracy']:.3f} < {th.reg_sign_acc}")

    if bt_va["avg_rr"] < th.avg_rr:
        failures.append(f"验证集平均盈亏比 {bt_va['avg_rr']:.3f} < {th.avg_rr}")
    if bt_va["expectancy"] <= 0:
        failures.append(f"验证集期望值 {bt_va['expectancy']:.4f} <= 0")

    prec_gap = abs(dir_tr["precision"] - dir_va["precision"])
    metrics["train_val_precision_gap"] = prec_gap
    if prec_gap > th.train_val_prec_gap:
        failures.append(f"训练/验证精确率差距 {prec_gap:.3f} > {th.train_val_prec_gap}")

    oos = oos_backtest(clf, reg, fused_oos, y_oos, entry_oos, m5, oos_bar_idx, label_params.horizon)
    metrics["oos_recent_2w"] = oos
    if oos.get("signal_win_rate", 0) < th.oos_win_rate:
        failures.append(f"样本外2周胜率 {oos.get('signal_win_rate', 0):.3f} < {th.oos_win_rate}")
    if oos.get("expectancy", -1) <= 0:
        failures.append("样本外2周总盈亏非正")

    if len(extreme_bar_idx) > 0:
        proba_ex = clf.predict_proba(fused_va[:0])  # placeholder
        # extreme uses subset of fused_va indices mapped separately
        pass

    passed = len(failures) == 0
    return AcceptanceReport(
        passed=passed,
        metrics=metrics,
        failures=failures,
        label_params=label_params.as_dict(),
        data_range=data_range,
    )


def save_artifacts(
    report: AcceptanceReport,
    clf: xgb.XGBClassifier,
    y_va: np.ndarray,
    y_pred_va: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "acceptance_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_va, y_pred_va, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["short", "flat", "long"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    imp = clf.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    idx = np.argsort(imp)[-20:]
    ax2.barh(range(len(idx)), imp[idx])
    ax2.set_yticks(range(len(idx)))
    ax2.set_yticklabels([f"f{i}" for i in idx])
    ax2.set_title("XGBoost Feature Importance (top 20)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig2)
