"""LightGBM v5 二分类训练（做多 vs 非多）。"""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from zhulong.training.lgb.acceptance import (
    LgbAcceptanceReport,
    evaluate_lgb_acceptance,
    get_thresholds,
)
from zhulong.training.lgb.backtest import DEFAULT_COOLDOWN_BARS, MAX_HOLD_BARS, backtest_signals
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB
from zhulong.training.lgb.labels_profit import DEFAULT_MAX_HOLD_BARS as PROFIT_MAX_HOLD
from zhulong.training.lgb.splits import DataSplits, split_indices

EARLY_STOPPING_ROUNDS = 50
NEG_RATIO = 5  # 正:负 = 1:5

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    n_signals: int
    signals_per_day: float = 0.0


def to_binary_long(labels: np.ndarray) -> np.ndarray:
    """1=做多, 0=观望或做空"""
    return (labels == 1).astype(int)


def _signals_per_day(times: pd.DatetimeIndex, n_signals: int) -> float:
    if n_signals == 0 or len(times) == 0:
        return 0.0
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    return n_signals / days


def _downsample(
    X: pd.DataFrame,
    y: np.ndarray,
    neg_ratio: int = NEG_RATIO,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0:
        return X.iloc[:0], y[:0]
    rng = np.random.default_rng(seed)
    n_neg = min(len(neg), len(pos) * neg_ratio)
    neg_pick = rng.choice(neg, size=n_neg, replace=False)
    ix = np.concatenate([pos, neg_pick])
    rng.shuffle(ix)
    return X.iloc[ix], y[ix]


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_signals": 0}
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_signals": int(y_pred.sum()),
    }


def threshold_sweep_binary(
    y_true: np.ndarray,
    proba: np.ndarray,
    val_times: pd.DatetimeIndex,
    lo: float = 0.30,
    hi: float = 0.90,
    step: float = 0.02,
) -> list[ThresholdResult]:
    rows: list[ThresholdResult] = []
    for thr in np.arange(lo, hi + step * 0.5, step):
        pred = (proba >= thr).astype(int)
        m = _clf_metrics(y_true, pred)
        rows.append(
            ThresholdResult(
                threshold=float(round(thr, 4)),
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                n_signals=m["n_signals"],
                signals_per_day=_signals_per_day(val_times, m["n_signals"]),
            )
        )
    return rows


def format_threshold_table(rows: list[ThresholdResult]) -> str:
    lines = ["Threshold | Precision | Recall | Signals/day", "-" * 50]
    for r in rows:
        lines.append(
            f"{r.threshold:.2f}      | {r.precision:.2f}      | {r.recall:.2f}   | {r.signals_per_day:.1f}"
        )
    return "\n".join(lines)


def tune_threshold_binary(
    y_true: np.ndarray,
    proba: np.ndarray,
    val_times: pd.DatetimeIndex,
    target_precision: float = 0.50,
    target_recall: float = 0.20,
    max_signals_per_day: float = 8.0,
    min_precision_fallback: float = 0.40,
    sweep_hi: float = 0.90,
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    rows = threshold_sweep_binary(y_true, proba, val_times, hi=sweep_hi)

    candidates = [
        r for r in rows
        if r.precision >= target_precision
        and r.signals_per_day <= max_signals_per_day
        and r.recall >= target_recall
        and r.n_signals > 0
    ]
    if candidates:
        return max(candidates, key=lambda r: (r.recall, r.precision)), rows

    candidates = [
        r for r in rows
        if r.precision >= target_precision
        and r.signals_per_day <= max_signals_per_day
        and r.n_signals > 0
    ]
    if candidates:
        return max(candidates, key=lambda r: (r.recall, r.precision)), rows

    candidates = [r for r in rows if r.signals_per_day <= max_signals_per_day and r.n_signals > 0]
    if candidates:
        return max(candidates, key=lambda r: (r.precision, r.recall)), rows

    candidates = [r for r in rows if r.precision >= min_precision_fallback and r.n_signals > 0]
    if candidates:
        return max(candidates, key=lambda r: (-r.signals_per_day, r.precision)), rows

    if rows:
        return max(rows, key=lambda r: r.precision), rows
    return ThresholdResult(0.50, 0, 0, 0, 0), rows


def _param_grid(quick: bool = False, v6: bool = False) -> list[dict[str, Any]]:
    if v6:
        base = {
            "num_leaves": [31, 63],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [500, 1000],
            "min_child_samples": [100, 200],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "reg_lambda": [1.0, 3.0],
        }
    else:
        base = {
            "num_leaves": [31, 63],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [500, 1000],
            "min_child_samples": [200, 500],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "reg_lambda": [3.0, 5.0],
        }
    keys = list(base.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[base[k] for k in keys])]
    random.shuffle(combos)
    return combos[: (8 if quick else 32)]


def train_binary_lgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    params: dict[str, Any],
) -> lgb.LGBMClassifier:
    p = dict(params)
    est = p.pop("n_estimators", 1000)
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=est,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **p,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return clf


def grid_search_binary(
    aligned: pd.DataFrame,
    splits: DataSplits,
    quick: bool = False,
    v6: bool = False,
    use_downsample: bool = True,
    target_precision: float = 0.50,
    target_recall: float = 0.20,
    max_signals_per_day: float = 8.0,
    sweep_hi: float = 0.90,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], ThresholdResult, list[ThresholdResult]]:
    tr_ix = splits.train.intersection(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    y_tr_full = to_binary_long(aligned.loc[tr_ix, "label"].values)
    y_va = to_binary_long(aligned.loc[va_ix, "label"].values)
    X_tr_full = aligned.loc[tr_ix, FEATURE_COLUMNS_LGB]
    X_va = aligned.loc[va_ix, FEATURE_COLUMNS_LGB]

    if use_downsample:
        X_tr, y_tr = _downsample(X_tr_full, y_tr_full, neg_ratio=NEG_RATIO)
        logger.info(
            "binary train pos=%s neg_sampled=%s (1:%s) val_pos=%s (%.2f%%)",
            y_tr_full.sum(), (y_tr == 0).sum(), NEG_RATIO,
            y_va.sum(), 100.0 * y_va.mean(),
        )
    else:
        X_tr, y_tr = X_tr_full, y_tr_full
        logger.info(
            "binary train (no downsample) pos=%s neg=%s val_pos=%s (%.2f%%)",
            y_tr.sum(), (y_tr == 0).sum(), y_va.sum(), 100.0 * y_va.mean(),
        )

    best_clf: lgb.LGBMClassifier | None = None
    best_cfg: dict[str, Any] = {}
    best_thr = ThresholdResult(0.50, 0, 0, 0, 0)
    best_rows: list[ThresholdResult] = []
    best_score = -1.0

    for cfg in _param_grid(quick, v6=v6):
        try:
            clf = train_binary_lgb(X_tr, y_tr, X_va, y_va, dict(cfg))
            proba = clf.predict_proba(X_va)[:, 1]
            thr_res, rows = tune_threshold_binary(
                y_va, proba, va_ix,
                target_precision=target_precision,
                target_recall=target_recall,
                max_signals_per_day=max_signals_per_day,
                sweep_hi=sweep_hi,
            )
            score = thr_res.precision * 0.45 + min(thr_res.recall / target_recall, 1.0) * 0.35
            score += max(0.0, 1.0 - thr_res.signals_per_day / max(max_signals_per_day, 1)) * 0.20
            if score > best_score:
                best_score = score
                best_clf = clf
                best_cfg = cfg
                best_thr = thr_res
                best_rows = rows
                logger.info(
                    "cfg=%s thr=%.2f prec=%.3f rec=%.3f sig/day=%.1f auc=%.4f",
                    cfg, thr_res.threshold, thr_res.precision, thr_res.recall,
                    thr_res.signals_per_day,
                    clf.best_score_.get("valid_0", {}).get("auc", 0.0)
                    if hasattr(clf, "best_score_") else 0.0,
                )
        except Exception as ex:
            logger.warning("cfg failed: %s", ex)

    if best_clf is None:
        raise RuntimeError("no valid binary LGB model")
    logger.info("threshold table:\n%s", format_threshold_table(best_rows))
    return best_clf, best_cfg, best_thr, best_rows


@dataclass
class BinaryTrainResult:
    symbol: str
    clf: lgb.LGBMClassifier
    threshold: ThresholdResult
    report: LgbAcceptanceReport
    model_cfg: dict[str, Any]
    label_horizon: int = 12
    gain_threshold: float = 0.0025


def run_binary_training(
    symbol: str,
    m5: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    quick: bool = False,
    acceptance_stage: str = "v5",
    label_horizon: int = 12,
    gain_threshold: float = 0.0025,
    target_precision: float = 0.50,
    target_recall: float = 0.20,
    max_signals_per_day: float = 8.0,
    cooldown_bars: int = 0,
    max_hold_bars: int = MAX_HOLD_BARS,
    use_downsample: bool | None = None,
) -> BinaryTrainResult:
    aligned = features.join(labels.rename("label"), how="inner")
    splits = split_indices(aligned.index)
    logger.info("split sizes: %s", {k: len(getattr(splits, k)) for k in ("train", "val", "test1", "stress")})

    is_profit = acceptance_stage in ("v6", "v61")
    if use_downsample is None:
        use_downsample = not is_profit
    if acceptance_stage == "v61":
        tp, min_rec, sweep_hi = 0.50, 0.15, 0.70
        max_hold = max_hold_bars
    elif acceptance_stage == "v6":
        tp, min_rec, sweep_hi = 0.55, 0.20, 0.90
        max_hold = max_hold_bars if max_hold_bars != MAX_HOLD_BARS else PROFIT_MAX_HOLD
    elif acceptance_stage == "v51":
        tp, min_rec, sweep_hi = 0.45, 0.10, 0.90
        max_hold = max_hold_bars
    else:
        tp, min_rec, sweep_hi = target_precision, target_recall, 0.90
        max_hold = max_hold_bars

    clf, cfg, thr, _ = grid_search_binary(
        aligned, splits, quick=quick, v6=is_profit, use_downsample=use_downsample,
        target_precision=tp,
        target_recall=min_rec,
        max_signals_per_day=max_signals_per_day,
        sweep_hi=sweep_hi,
    )

    va_ix = splits.val.intersection(aligned.index)
    X_va = aligned.loc[va_ix, FEATURE_COLUMNS_LGB]
    y_va = to_binary_long(aligned.loc[va_ix, "label"].values)
    proba_va = clf.predict_proba(X_va)[:, 1]
    pred_va = (proba_va >= thr.threshold).astype(int)
    val_cls = {
        "precision": float(precision_score(y_va, pred_va, zero_division=0)),
        "recall": float(recall_score(y_va, pred_va, zero_division=0)),
        "f1": float(f1_score(y_va, pred_va, zero_division=0)),
        "n_signals": int(pred_va.sum()),
        "signals_per_day": _signals_per_day(va_ix, int(pred_va.sum())),
        "auc": float(clf.best_score_.get("valid_0", {}).get("auc", 0.0) if hasattr(clf, "best_score_") else 0.0),
    }

    test_ix = splits.test1.intersection(aligned.index)
    proba_test = clf.predict_proba(aligned.loc[test_ix, FEATURE_COLUMNS_LGB])[:, 1]
    dirs_test = np.where(proba_test >= thr.threshold, 1, 0)
    stress_ix = splits.stress.intersection(aligned.index)
    proba_stress = clf.predict_proba(aligned.loc[stress_ix, FEATURE_COLUMNS_LGB])[:, 1]
    dirs_stress = np.where(proba_stress >= thr.threshold, 1, 0)
    bt_cooldown = cooldown_bars if cooldown_bars > 0 else (
        DEFAULT_COOLDOWN_BARS if acceptance_stage in ("v51", "v6", "v61") else 0
    )
    test1_bt = backtest_signals(
        m5, test_ix, dirs_test, max_hold=max_hold, cooldown_bars=bt_cooldown,
    )
    stress_bt = backtest_signals(
        m5, stress_ix, dirs_stress, max_hold=max_hold, cooldown_bars=bt_cooldown,
    )

    report = evaluate_lgb_acceptance(
        val_cls, test1_bt, stress_bt,
        thresholds=get_thresholds(acceptance_stage),
        stage=acceptance_stage,
    )
    report.metrics["threshold"] = thr.__dict__

    return BinaryTrainResult(
        symbol=symbol,
        clf=clf,
        threshold=thr,
        report=report,
        model_cfg=cfg,
        label_horizon=label_horizon,
        gain_threshold=gain_threshold,
    )


def save_binary_model(
    result: BinaryTrainResult,
    out_dir: Path,
    model_name: str = "lgb_binary.txt",
    meta_name: str = "lgb_binary_meta.pkl",
    config_name: str = "config_v5.json",
    manifest_name: str = "manifest_v5.json",
    version: str = "v5",
    cooldown_bars: int = 0,
    max_hold_bars: int = MAX_HOLD_BARS,
    label_mode: str = "binary_long",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.clf.booster_.save_model(str(out_dir / model_name))
    joblib.dump(
        {
            "threshold": result.threshold.threshold,
            "feature_columns": FEATURE_COLUMNS_LGB,
            "label_horizon": result.label_horizon,
            "gain_threshold": result.gain_threshold,
            "max_hold_bars": max_hold_bars,
            "cooldown_bars": cooldown_bars,
            "mode": label_mode,
        },
        out_dir / meta_name,
    )
    manifest = {
        "symbol": result.symbol,
        "kind": "production" if result.report.passed else "rejected",
        "acceptance_passed": result.report.passed,
        "acceptance_stage": result.report.acceptance_stage,
        "classifier_mode": "lgb_binary_long",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": result.report.to_dict(),
        "model_cfg": result.model_cfg,
    }
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "version": version,
        "horizon": result.label_horizon,
        "gain_threshold": result.gain_threshold,
        "threshold": result.threshold.threshold,
        "max_hold_bars": max_hold_bars,
        "cooldown_bars": cooldown_bars,
        "neg_ratio": NEG_RATIO if label_mode != "profit_long" else 1,
        "label_mode": label_mode,
        "model_cfg": result.model_cfg,
    }
    (out_dir / config_name).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
