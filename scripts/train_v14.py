#!/usr/bin/env python3
"""
V14: 修复列映射 bug + 优化参数 + 多验证窗口 + 合理验收标准。

核心修复：
  1. to_multiclass 输出 0=flat/1=long/2=short（与 proba column 一致）
  2. gain 阈值 0.20%（减少标签噪声）
  3. 更强的 XGBoost 正则化（防过拟合）
  4. 三段验证（2023 评估，2024 优化阈值，2025 最终验收）
  5. 简化后处理（移除 H1 EMA50 趋势过滤，保留 ATR）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score

from zhulong.training.lgb.acceptance import (
    LgbAcceptanceReport,
    evaluate_lgb_acceptance,
    thresholds_v13_triple,
)
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.labels import generate_direction_labels
from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.v10.backtest import MIN_ATR_PCT, backtest_both

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 时间切分：三段验证 (2023 / 2024 / 2025)
# ---------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
VAL2023_START = pd.Timestamp("2023-01-01 00:00:00")
VAL2023_END = pd.Timestamp("2023-12-31 23:59:59")
VAL2024_START = pd.Timestamp("2024-01-01 00:00:00")
VAL2024_END = pd.Timestamp("2024-12-31 23:59:59")
TEST2025_START = pd.Timestamp("2025-01-01 00:00:00")
TEST2025_END = pd.Timestamp("2025-12-31 23:59:59")

# ---------------------------------------------------------------------------
# 标签参数（提升 gain 阈值减少噪声）
# ---------------------------------------------------------------------------
V14_HORIZON = 12
V14_GAIN = 0.0020  # 0.20%（vs v13 的 0.15%）

# ---------------------------------------------------------------------------
# 回测参数
# ---------------------------------------------------------------------------
V14_MAX_HOLD = 12
V14_COOLDOWN = 12  # 缩短冷却（vs v13 的 18）
V14_MAX_DAILY = 8

# ---------------------------------------------------------------------------
# XGBoost 优化参数（更强正则化 + 更多树 + 更低学习率）
# ---------------------------------------------------------------------------
V14_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_lambda": 8.0,
    "reg_alpha": 2.0,
    "min_child_weight": 7,
    "gamma": 0.2,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
}

def _detect_gpu_params() -> dict:
    """检测 CUDA GPU 并返回 XGBoost GPU 加速参数。"""
    try:
        import xgboost as xgb
        # xgboost 3.x: 仅用 device='cuda' 即可启用 GPU 训练
        test = xgb.XGBClassifier(device="cuda", n_estimators=1, max_depth=2)
        test.fit([[0]], [0], verbose=False)
        logger.info("CUDA GPU detected — using device=cuda")
        return {"device": "cuda"}
    except Exception:
        logger.warning("GPU not available — falling back to CPU")
        return {"n_jobs": -1}


# ===================================================================
# 修复：正确列映射的 to_multiclass 和 proba_to_directions
# ===================================================================

def to_multiclass_v14(labels: np.ndarray) -> np.ndarray:
    """
    -1/0/1 → 0/1/2
    列 0 = flat, 列 1 = long, 列 2 = short
    
    这是 V14 的核心修复：之前的 to_multiclass 将 -1 映射到 col 0 (short),
    但 proba_to_directions 假设 col 0 是 flat，导致做多/做空逻辑完全颠倒。
    """
    out = np.zeros(len(labels), dtype=int)  # default: 0 = flat
    out[labels == 1] = 1   # col 1 = long
    out[labels == -1] = 2  # col 2 = short
    return out


def proba_to_directions_v14(
    proba: np.ndarray,
    long_thr: float,
    short_thr: float,
) -> np.ndarray:
    """
    proba columns: 0=flat, 1=long, 2=short
    
    与上面 to_multiclass_v14 的列映射保持一致。
    """
    n = len(proba)
    dirs = np.zeros(n, dtype=np.int8)
    p_flat = proba[:, 0]
    p_long = proba[:, 1]
    p_short = proba[:, 2]
    
    for i in range(n):
        # 做多条件：long 概率 ≥ 阈值 且是最高概率 且 > flat 概率
        if p_long[i] >= long_thr and p_long[i] >= p_short[i] and p_long[i] > p_flat[i]:
            dirs[i] = 1
        # 做空条件：short 概率 ≥ 阈值 且是最高概率 且 > flat 概率
        elif p_short[i] >= short_thr and p_short[i] >= p_long[i] and p_short[i] > p_flat[i]:
            dirs[i] = -1
    return dirs


# ===================================================================
# ATR 过滤（保留，但阈值放宽到 0.05%）
# ===================================================================

MIN_ATR_PCT_V14 = 0.0005  # 0.05%（vs v13 的 0.10%）


def atr_filter(m5: pd.DataFrame, times: pd.DatetimeIndex, dirs: np.ndarray) -> np.ndarray:
    """ATR 低波动过滤 —— 剔除极低波动假信号。"""
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
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT_V14:
            out[i] = 0
    return out


# ===================================================================
# 类权重
# ===================================================================

def class_sample_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float64)
    for label in (0, 1, 2):
        cnt = max(int((y == label).sum()), 1)
        w = len(y) / (3.0 * cnt)
        # 多空方向额外加权 1.5×
        if label in (1, 2):
            w *= 1.5
        weights[y == label] = w
    return weights


# ===================================================================
# 非对称阈值扫描
# ===================================================================

def tune_thresholds_v14(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    lo: float = 0.40,
    hi: float = 0.60,
    step: float = 0.02,
    target_precision: float = 0.55,
    max_signals_per_day: float = 8.0,
    min_signals_per_day: float = 0.0,
    min_signals: int = 100,
) -> tuple[dict, list[dict]]:
    """多空独立阈值扫描，选加权精确率最高的组合。"""
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    best: dict | None = None
    best_score = -1.0
    rows: list[dict] = []
    
    for long_thr in np.arange(lo, hi + step * 0.5, step):
        for short_thr in np.arange(lo, hi + step * 0.5, step):
            dirs = atr_filter(m5, times, proba_to_directions_v14(proba, long_thr, short_thr))
            pred_cls = np.zeros(len(dirs), dtype=int)
            pred_cls[dirs == 1] = 1
            pred_cls[dirs == -1] = 2
            mask = pred_cls > 0
            n_sig = int(mask.sum())
            if n_sig < min_signals:
                continue
            spd = n_sig / days
            if spd > max_signals_per_day:
                continue
            if spd < min_signals_per_day:
                continue
            
            long_m = dirs == 1
            short_m = dirs == -1
            n_long = int(long_m.sum())
            n_short = int(short_m.sum())
            lp = float((y_true[long_m] == 1).mean()) if n_long > 0 else 0.0
            sp = float((y_true[short_m] == 2).mean()) if n_short > 0 else 0.0
            wprec = (lp * n_long + sp * n_short) / max(n_sig, 1)
            
            row = {
                "long_threshold": float(round(long_thr, 2)),
                "short_threshold": float(round(short_thr, 2)),
                "weighted_precision": wprec,
                "long_precision": lp,
                "short_precision": sp,
                "n_signals": n_sig,
                "n_long": n_long,
                "n_short": n_short,
                "signals_per_day": spd,
            }
            rows.append(row)
            
            # 评分：精确率优先，信号数量为辅
            if lp >= target_precision and sp >= target_precision:
                score = wprec + min(n_sig / 200.0, 1.0) * 0.01
                if score > best_score:
                    best_score = score
                    best = row
    
    if best is None:
        # 放宽：至少一侧 ≥ target_precision
        qualified = [r for r in rows if r["long_precision"] >= target_precision or r["short_precision"] >= target_precision]
        if qualified:
            best = max(qualified, key=lambda x: (x["weighted_precision"], x["n_signals"]))
        elif rows:
            best = max(rows, key=lambda x: (x["weighted_precision"], x["n_signals"]))
        else:
            best = {"long_threshold": 0.50, "short_threshold": 0.50, "weighted_precision": 0.0,
                    "long_precision": 0.0, "short_precision": 0.0, "n_signals": 0,
                    "n_long": 0, "n_short": 0, "signals_per_day": 0.0}
    
    return best, rows


# ===================================================================
# 主训练流程
# ===================================================================

@dataclass
class V14TrainResult:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    long_threshold: float
    short_threshold: float
    report: object
    clf_report: str


def run_v14_training(
    symbol: str = "XAUUSD",
    horizon: int = V14_HORIZON,
    gain: float = V14_GAIN,
    quick: bool = False,
) -> V14TrainResult:
    root = _ROOT
    m5_path = root / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv"
    if not m5_path.is_file():
        m5_path = root / "data" / "training" / f"{symbol}_M5.csv"
    m5 = load_vendor_csv(m5_path)
    logger.info("M5 loaded: %d bars, %s ~ %s", len(m5), m5.index[0], m5.index[-1])

    # 特征计算（复用 v13 的特征缓存）
    feat_cache = root / "data" / "training" / "v14" / symbol / "features.parquet"
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
        logger.info("loaded cached features: %s rows", len(feats))
    else:
        feats = compute_features(m5, include_reversal=True)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)
        logger.info("features cached: %s", feat_cache)

    # 标签：提高 gain 阈值减少噪声
    raw_labels = generate_direction_labels(m5.loc[feats.index], horizon=horizon, gain_threshold=gain)
    labels = pd.Series(to_multiclass_v14(raw_labels), index=feats.index, name="label")

    label_dist = {int(v): int((labels.values == v).sum()) for v in (0, 1, 2)}
    logger.info("labels: flat(0)=%d long(1)=%d short(2)=%d", label_dist[0], label_dist[1], label_dist[2])

    aligned = feats.join(labels, how="inner").dropna(subset=["label"])
    cols = list(FEATURE_COLUMNS_LGB_V13)

    # 三段切分
    train_ix = aligned.index[aligned.index <= TRAIN_END]
    val23_ix = aligned.index[
        (aligned.index >= VAL2023_START) & (aligned.index <= VAL2023_END)
    ]
    val24_ix = aligned.index[
        (aligned.index >= VAL2024_START) & (aligned.index <= VAL2024_END)
    ]
    test25_ix = aligned.index[
        (aligned.index >= TEST2025_START) & (aligned.index <= TEST2025_END)
    ]
    logger.info(
        "split: train=%d val23=%d val24=%d test25=%d",
        len(train_ix), len(val23_ix), len(val24_ix), len(test25_ix),
    )

    X_tr = aligned.loc[train_ix, cols]
    y_tr = aligned.loc[train_ix, "label"].values.astype(int)
    X_v23 = aligned.loc[val23_ix, cols]
    y_v23 = aligned.loc[val23_ix, "label"].values.astype(int)
    X_v24 = aligned.loc[val24_ix, cols]
    y_v24 = aligned.loc[val24_ix, "label"].values.astype(int)

    sw = class_sample_weights(y_tr)

    # 训练 — 自动检测 GPU
    gpu_params = _detect_gpu_params()
    n_est = 150 if quick else V14_XGB_PARAMS["n_estimators"]
    model_params = {**V14_XGB_PARAMS, **gpu_params, "n_estimators": n_est}
    model = xgb.XGBClassifier(**model_params)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_v23, y_v23)],
        verbose=50 if not quick else False,
    )

    # 评估：2023 验证集分类报告
    y_pred23 = model.predict(X_v23)
    clf_rep = classification_report(
        y_v23, y_pred23,
        target_names=["flat", "long", "short"],
        zero_division=0,
    )
    logger.info("=== Val 2023 classification ===\n%s", clf_rep)

    # 在 2024 验证集上优化阈值
    proba_v24 = model.predict_proba(X_v24)
    best_thr, sweep = tune_thresholds_v14(
        proba_v24, y_v24, val24_ix, m5,
        lo=0.40, hi=0.70, step=0.02,
        target_precision=0.55,
        max_signals_per_day=V14_MAX_DAILY,
        min_signals_per_day=2.0,
        min_signals=50,
    )
    long_thr = best_thr["long_threshold"]
    short_thr = best_thr["short_threshold"]
    logger.info(
        "[V14] thresholds: long=%.2f short=%.2f wprec=%.3f n_long=%d n_short=%d sig/day=%.1f",
        long_thr, short_thr,
        best_thr["weighted_precision"],
        best_thr.get("n_long", 0), best_thr.get("n_short", 0),
        best_thr["signals_per_day"],
    )

    # 在 2024 验证集上计算最终指标
    dirs_v24 = atr_filter(m5, val24_ix, proba_to_directions_v14(proba_v24, long_thr, short_thr))
    pred_cls_v24 = np.zeros(len(dirs_v24), dtype=int)
    pred_cls_v24[dirs_v24 == 1] = 1
    pred_cls_v24[dirs_v24 == -1] = 2
    sig_mask_24 = pred_cls_v24 > 0

    val_metrics = {
        "precision": best_thr["weighted_precision"],
        "recall": float(
            recall_score(y_v24[sig_mask_24], pred_cls_v24[sig_mask_24],
                        labels=[1, 2], average="macro", zero_division=0)
        ) if sig_mask_24.any() else 0.0,
        "long_precision": best_thr["long_precision"],
        "short_precision": best_thr["short_precision"],
        "n_signals": int(sig_mask_24.sum()),
    }

    # 2025 样本外回测
    X_t25 = aligned.loc[test25_ix, cols]
    proba_t25 = model.predict_proba(X_t25)
    dirs_t25 = atr_filter(m5, test25_ix, proba_to_directions_v14(proba_t25, long_thr, short_thr))
    oos_bt = backtest_both(
        m5, test25_ix, dirs_t25,
        max_hold=V14_MAX_HOLD,
        cooldown_bars=V14_COOLDOWN,
        max_daily_signals=V14_MAX_DAILY,
    )
    logger.info("[V14] OOS 2025 backtest: %s", json.dumps(oos_bt, indent=2, default=str))

    # 验收
    report = evaluate_lgb_acceptance(val_metrics, oos_bt, {}, stage="v13_triple")
    report.metrics["thresholds"] = best_thr
    report.metrics["threshold_sweep"] = sweep
    report.metrics["horizon"] = horizon
    report.metrics["gain"] = gain

    # 附加检查
    extra_failures: list[str] = []
    if best_thr["long_precision"] < 0.50:
        extra_failures.append(f"long_precision={best_thr['long_precision']:.3f}<0.50")
    if best_thr["short_precision"] < 0.50:
        extra_failures.append(f"short_precision={best_thr['short_precision']:.3f}<0.50")
    n_trades = int(oos_bt.get("n_trades", 0))
    if n_trades < 100:
        extra_failures.append(f"n_trades={n_trades}<100 (信号过少)")
    if oos_bt.get("max_drawdown", 1) > 0.30:
        extra_failures.append(f"max_drawdown={oos_bt['max_drawdown']:.3f}>0.30")
    if extra_failures:
        report.failures.extend(extra_failures)
        report.passed = False

    return V14TrainResult(model, cols, long_thr, short_thr, report, clf_rep)


def save_v14_model(result: V14TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_v14.json"))
    meta = {
        "params": V14_XGB_PARAMS,
        "feature_columns": result.feature_columns,
        "long_threshold": result.long_threshold,
        "short_threshold": result.short_threshold,
        "max_hold_bars": V14_MAX_HOLD,
        "cooldown_bars": V14_COOLDOWN,
        "max_daily_signals": V14_MAX_DAILY,
        "horizon": V14_HORIZON,
        "gain": V14_GAIN,
        "column_mapping": {"flat": 0, "long": 1, "short": 2},
    }
    joblib.dump(meta, out_dir / "v14_meta.pkl")
    (out_dir / "feature_columns.json").write_text(
        json.dumps(result.feature_columns, indent=2), encoding="utf-8",
    )
    (out_dir / "config_v14.json").write_text(json.dumps(
        {
            "long_threshold": result.long_threshold,
            "short_threshold": result.short_threshold,
            "passed": result.report.passed,
            "n_features": len(result.feature_columns),
        },
        indent=2,
    ), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 训练（修复列映射 + 优化参数）")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--horizon", type=int, default=V14_HORIZON)
    parser.add_argument("--gain", type=float, default=V14_GAIN)
    parser.add_argument("--quick", action="store_true", help="快速试跑（150 棵树）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = run_v14_training(
        symbol=args.symbol,
        horizon=args.horizon,
        gain=args.gain,
        quick=args.quick,
    )

    out_dir = _ROOT / "models" / args.symbol / "v14"
    save_v14_model(result, out_dir)

    report_dir = _ROOT / "data" / "training" / "reports" / "v14" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_report_v14.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "classification_report_v14.txt").write_text(
        result.clf_report, encoding="utf-8",
    )

    logger.info("=== V14 验收结论 ===")
    logger.info("passed=%s", result.report.passed)
    if result.report.failures:
        logger.error("failures: %s", result.report.failures)
    else:
        logger.info("ALL CHECKS PASSED!")
    
    oos = result.report.metrics.get("test1", {})
    logger.info(
        "OOS: win_rate=%.1f%% avg_rr=%.2f n_trades=%d max_dd=%.1f%% total_pnl=%.1fR",
        oos.get("win_rate", 0) * 100,
        oos.get("avg_rr", 0),
        oos.get("n_trades", 0),
        oos.get("max_drawdown", 0) * 100,
        oos.get("total_pnl_r", 0),
    )

    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
