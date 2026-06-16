#!/usr/bin/env python3
"""Optuna 超参数优化（TimeSeriesSplit + 多分类 logloss）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from zhulong.training.v12.train import boost_short_samples
from zhulong.training.v13.train_pipeline import load_training_bundle
from zhulong.training.v13.triple import class_sample_weights

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    losses: list[float] = []
    for tr_idx, va_idx in tscv.split(X):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            sample_weight=sample_weight[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            verbose=False,
        )
        proba = model.predict_proba(X[va_idx])
        y_va = y[va_idx]
        ll = -np.mean(np.log(np.clip(proba[np.arange(len(y_va)), y_va], 1e-15, 1.0)))
        losses.append(float(ll))
        trial.report(float(np.mean(losses)), len(losses))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--model", choices=["xgb", "lgb"], default="xgb")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bundle = load_training_bundle(_ROOT, args.symbol, include_enhanced=args.enhanced)
    train_bal = boost_short_samples(bundle["train_bal"], short_mult=1)
    cols = bundle["cols"]
    X = train_bal[cols].to_numpy(dtype=np.float32)
    y = train_bal["label"].values.astype(int)
    sw = class_sample_weights(y)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(lambda t: objective(t, X, y, sw), n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    out_dir = _ROOT / "data" / "training" / "reports" / "optuna" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"best_params": best, "best_value": study.best_value, "n_trials": len(study.trials)}
    (out_dir / "best_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("best logloss=%.4f params=%s", study.best_value, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
