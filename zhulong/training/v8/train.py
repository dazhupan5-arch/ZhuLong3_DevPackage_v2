"""v8 训练：XGBoost 回归 + LightGBM 分类 + 软投票集成。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance, get_thresholds
from zhulong.training.lgb.backtest import DEFAULT_COOLDOWN_BARS, backtest_signals
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train_binary import (
    ThresholdResult,
    format_threshold_table,
    tune_threshold_binary,
)

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "min_child_samples": 100,
    "verbose": -1,
    "random_state": 42,
}


@dataclass
class V8TrainResult:
    xgb_model: xgb.XGBRegressor
    lgb_model: lgb.LGBMClassifier
    threshold: ThresholdResult
    feature_columns: list[str]
    report: Any
    xgb_weight: float = 0.5


def _split_xy(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    cols: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    aligned = feats.join(labels, how="inner")
    splits = split_indices(aligned.index)
    out: dict[str, tuple] = {}
    for name in ("train", "val", "test1"):
        ix = getattr(splits, name).intersection(aligned.index)
        sub = aligned.loc[ix]
        out[name] = (
            sub[cols].to_numpy(dtype=np.float32),
            sub["label_reg"].to_numpy(dtype=np.float32),
            sub["label_cls"].to_numpy(dtype=np.int8),
            ix,
        )
    return out


def train_v8_models(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    m5: pd.DataFrame,
    quick: bool = False,
) -> V8TrainResult:
    data = _split_xy(feats, labels, feature_columns)
    X_tr, y_reg_tr, y_cls_tr, _ = data["train"]
    X_va, y_reg_va, y_cls_va, va_ix = data["val"]
    X_te, _, y_cls_te, te_ix = data["test1"]

    n_est = 100 if quick else XGB_PARAMS["n_estimators"]

    xgb_model = xgb.XGBRegressor(**{**XGB_PARAMS, "n_estimators": n_est})
    xgb_model.fit(
        X_tr,
        y_reg_tr,
        eval_set=[(X_va, y_reg_va)],
        verbose=False,
    )

    pos = max(int(y_cls_tr.sum()), 1)
    neg = max(int((y_cls_tr == 0).sum()), 1)
    scale = (neg / pos) * 0.5 if pos / len(y_cls_tr) < 0.2 else 1.0

    lgb_model = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_est, "scale_pos_weight": scale})
    lgb_model.fit(
        X_tr,
        y_cls_tr,
        eval_set=[(X_va, y_cls_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred_xgb_va = xgb_model.predict(X_va)
    pred_lgb_va = lgb_model.predict_proba(X_va)[:, 1]
    xgb_norm = pred_xgb_va / max(np.abs(pred_xgb_va).max(), 1e-9)
    xgb_prob = 1.0 / (1.0 + np.exp(-xgb_norm * 10))
    final_prob = 0.5 * xgb_prob + 0.5 * pred_lgb_va

    thr, rows = tune_threshold_binary(
        y_cls_va.astype(int),
        final_prob,
        va_ix,
        target_precision=0.55,
        target_recall=0.10,
        max_signals_per_day=8.0,
        sweep_hi=0.80,
    )
    logger.info("ensemble threshold table:\n%s", format_threshold_table(rows))

    pred_xgb_te = xgb_model.predict(X_te)
    pred_lgb_te = lgb_model.predict_proba(X_te)[:, 1]
    xgb_norm_te = pred_xgb_te / max(np.abs(pred_xgb_te).max(), 1e-9)
    xgb_prob_te = 1.0 / (1.0 + np.exp(-xgb_norm_te * 10))
    final_te = 0.5 * xgb_prob_te + 0.5 * pred_lgb_te
    dirs_te = np.where(final_te >= thr.threshold, 1, 0)

    val_pred = (final_prob >= thr.threshold).astype(int)
    val_cls = {
        "precision": float(precision_score(y_cls_va, val_pred, zero_division=0)),
        "recall": float(recall_score(y_cls_va, val_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_cls_va, final_prob)),
        "n_signals": int(val_pred.sum()),
    }

    test1_bt = backtest_signals(
        m5, te_ix, dirs_te, max_hold=24, cooldown_bars=DEFAULT_COOLDOWN_BARS,
    )
    report = evaluate_lgb_acceptance(val_cls, test1_bt, {}, stage="v8")
    report.metrics["threshold"] = thr.__dict__

    return V8TrainResult(
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        threshold=thr,
        feature_columns=feature_columns,
        report=report,
    )


def save_v8_models(result: V8TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.xgb_model.save_model(str(out_dir / "xgb_regressor.json"))
    result.lgb_model.booster_.save_model(str(out_dir / "lgb_classifier.txt"))
    joblib.dump(
        {
            "feature_columns": result.feature_columns,
            "threshold": result.threshold.threshold,
            "xgb_weight": result.xgb_weight,
            "max_hold_bars": 24,
            "cooldown_bars": DEFAULT_COOLDOWN_BARS,
        },
        out_dir / "v8_meta.pkl",
    )
    (out_dir / "config_v8.json").write_text(
        json.dumps(
            {
                "threshold": result.threshold.threshold,
                "n_features": len(result.feature_columns),
                "passed": result.report.passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
