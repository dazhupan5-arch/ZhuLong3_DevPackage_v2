#!/usr/bin/env python3
"""v13 三重屏障标签 + 68 维特征 + 下采样训练。"""

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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.backtest import SL_ATR, TP_ATR
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v10.backtest import backtest_both
from zhulong.training.v11.train import V11_COOLDOWN, V11_MAX_HOLD, proba_to_directions
from zhulong.training.v12.train import boost_short_samples
from zhulong.training.v13.triple import (
    TEST_END,
    TEST_START,
    V13_TRIPLE_XGB,
    class_sample_weights,
    postprocess_directions,
    tune_precision_thresholds,
)

V13_MAX_DAILY = 6

logger = logging.getLogger(__name__)


@dataclass
class V13TripleResult:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    long_threshold: float
    short_threshold: float
    report: object
    clf_report: str


def run_v13_triple_training(
    symbol: str = "XAUUSD",
    labels_path: str = "data/training/XAUUSD_triple_v3.csv",
    train_balanced_path: str = "data/training/train_balanced_v3.csv",
    out_subdir: str = "triple_barrier",
    model_name: str = "xgb_triple_v3.json",
    quick: bool = False,
    short_mult: int = 1,
) -> V13TripleResult:
    root = _ROOT
    m5_path = root / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv"
    m5 = load_vendor_csv(m5_path)

    feat_cache = root / "data" / "training" / "v13" / symbol / "features.parquet"
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
        logger.info("loaded cached features: %s", feat_cache)
    else:
        feats = compute_features(m5, include_reversal=True)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)

    lab = pd.read_csv(root / labels_path, index_col=0, parse_dates=True)
    aligned = feats.join(lab[["label"]], how="inner").dropna(subset=["label"])
    aligned["label"] = aligned["label"].astype(int)
    cols = list(FEATURE_COLUMNS_LGB_V13)

    splits = split_indices(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    te_ix = aligned.index[(aligned.index >= TEST_START) & (aligned.index <= TEST_END)]

    train_bal = pd.read_csv(root / train_balanced_path)
    if short_mult > 1:
        train_bal = boost_short_samples(train_bal, short_mult=short_mult)
    X_tr = train_bal[cols]
    y_tr = train_bal["label"].values.astype(int)
    X_va = aligned.loc[va_ix, cols]
    y_va = aligned.loc[va_ix, "label"].values.astype(int)

    logger.info("train=%d val=%d test2025=%d", len(y_tr), len(y_va), len(te_ix))

    n_est = 150 if quick else V13_TRIPLE_XGB["n_estimators"]
    model = xgb.XGBClassifier(**{**V13_TRIPLE_XGB, "n_estimators": n_est})
    model.fit(
        X_tr,
        y_tr,
        sample_weight=class_sample_weights(y_tr),
        eval_set=[(X_va, y_va)],
        verbose=50 if not quick else False,
    )

    y_pred = model.predict(X_va)
    clf_rep = classification_report(y_va, y_pred, target_names=["flat", "long", "short"], zero_division=0)
    logger.info("val clf:\n%s", clf_rep)

    proba_va = model.predict_proba(X_va)
    long_thr, short_thr, sweep, thr_m = tune_precision_thresholds(
        proba_va, y_va, va_ix, m5, target_precision=0.55, max_signals_per_day=V13_MAX_DAILY
    )
    logger.info(
        "threshold=%.2f wprec=%.3f long=%.3f short=%.3f sig/day=%.1f",
        long_thr,
        thr_m.get("weighted_precision", 0),
        thr_m.get("long_precision", 0),
        thr_m.get("short_precision", 0),
        thr_m.get("signals_per_day", 0),
    )

    dirs_va = postprocess_directions(m5, va_ix, proba_va, long_thr, short_thr)
    pred_cls = np.zeros(len(dirs_va), dtype=int)
    pred_cls[dirs_va == 1] = 1
    pred_cls[dirs_va == -1] = 2
    sig_mask = pred_cls > 0
    trade_recall = float(
        recall_score(y_va[sig_mask], pred_cls[sig_mask], labels=[1, 2], average="macro", zero_division=0)
    ) if sig_mask.any() else 0.0

    val_metrics = {
        "precision": thr_m.get("weighted_precision", 0.0),
        "recall": trade_recall,
        "long_precision": thr_m.get("long_precision", 0.0),
        "short_precision": thr_m.get("short_precision", 0.0),
        "trade_f1": thr_m.get("trade_f1", 0.0),
        "n_signals": int(sig_mask.sum()),
    }

    proba_te = model.predict_proba(aligned.loc[te_ix, cols])
    dirs_te = postprocess_directions(m5, te_ix, proba_te, long_thr, short_thr)
    oos_bt = backtest_both(
        m5, te_ix, dirs_te,
        max_hold=V11_MAX_HOLD,
        cooldown_bars=V11_COOLDOWN,
        max_daily_signals=V13_MAX_DAILY,
    )
    logger.info("oos 2025: %s", oos_bt)

    report = evaluate_lgb_acceptance(val_metrics, oos_bt, {}, stage="v13_triple")
    report.metrics["thresholds"] = {"long_thr": long_thr, "short_thr": short_thr, **thr_m}
    report.metrics["threshold_sweep"] = sweep
    report.metrics["label_sl_tp"] = {"sl": SL_ATR, "tp_train": TP_ATR, "tp_backtest": TP_ATR}
    report.metrics["trend_filter"] = "h1_ema50_slope"

    failures_extra: list[str] = []
    if thr_m.get("long_precision", 0) < 0.55:
        failures_extra.append(f"long_precision={thr_m.get('long_precision', 0):.3f}<0.55")
    if thr_m.get("short_precision", 0) < 0.55:
        failures_extra.append(f"short_precision={thr_m.get('short_precision', 0):.3f}<0.55")
    if trade_recall < 0.30:
        failures_extra.append(f"trade_recall={trade_recall:.3f}<0.30")
    n_trades = int(oos_bt.get("n_trades", 0))
    if n_trades < 200:
        failures_extra.append(f"n_trades={n_trades}<200")
    if n_trades > 400:
        failures_extra.append(f"n_trades={n_trades}>400")
    if oos_bt.get("max_drawdown", 1) > 0.20:
        failures_extra.append(f"max_drawdown={oos_bt.get('max_drawdown', 1):.3f}>0.20")
    if failures_extra:
        report.failures.extend(failures_extra)
        report.passed = False

    return V13TripleResult(model, cols, long_thr, short_thr, report, clf_rep)


def save_v13_triple_model(
    result: V13TripleResult,
    out_dir: Path,
    model_name: str = "xgb_triple_v3.json",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / model_name))
    result.model.save_model(str(out_dir / "xgb_direction.json"))
    joblib.dump(
        {
            "params": V13_TRIPLE_XGB,
            "feature_columns": result.feature_columns,
            "long_threshold": result.long_threshold,
            "short_threshold": result.short_threshold,
            "label_mode": "triple_barrier",
            "max_hold_bars": V11_MAX_HOLD,
        },
        out_dir / "params_v13_triple.pkl",
    )
    (out_dir / "feature_columns.json").write_text(
        json.dumps(result.feature_columns, indent=2), encoding="utf-8"
    )
    (out_dir / "config_v13_triple.json").write_text(
        json.dumps(
            {
                "long_threshold": result.long_threshold,
                "short_threshold": result.short_threshold,
                "passed": result.report.passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_acceptance_report(result: V13TripleResult, symbol: str) -> Path:
    report_dir = _ROOT / "data" / "training" / "reports" / "v13_triple" / symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    jpath = report_dir / "acceptance_report_v13_triple.json"
    jpath.write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "classification_report.txt").write_text(result.clf_report, encoding="utf-8")

    val = result.report.metrics.get("validation", {})
    oos = result.report.metrics.get("test1", {})
    thr = result.report.metrics.get("thresholds", {})
    md = [
        "# V13 Triple Barrier v3 验收报告",
        "",
        "> 标签/回测统一 SL=1.2 TP=2.0 | 下采样 1:1:2 | 多空权重 2× | H1 EMA50 趋势过滤",
        "",
        f"**结论**: {'PASS' if result.report.passed else 'FAIL'}",
        "",
        "## 验证集",
        f"| 指标 | 结果 | 目标 |",
        f"|------|------|------|",
        f"| 加权精确率 | {val.get('precision', 0):.1%} | ≥ 55% |",
        f"| 做多精确率 | {val.get('long_precision', 0):.1%} | ≥ 55% |",
        f"| 做空精确率 | {val.get('short_precision', 0):.1%} | ≥ 55% |",
        f"| 交易召回率 | {val.get('recall', 0):.1%} | ≥ 30% |",
        f"| 做多阈值 | {thr.get('long_threshold', thr.get('long_thr', 0)):.2f} | — |",
        f"| 做空阈值 | {thr.get('short_threshold', thr.get('short_thr', 0)):.2f} | — |",
        "",
        f"## 2025 样本外（SL={SL_ATR}×ATR / TP={TP_ATR}×ATR）",
        f"| 指标 | 结果 | 目标 |",
        f"|------|------|------|",
        f"| 胜率 | {oos.get('win_rate', 0):.1%} | ≥ 55% |",
        f"| 盈亏比 | {oos.get('avg_rr', 0):.2f} | ≥ 1.5 |",
        f"| 交易笔数 | {oos.get('n_trades', 0)} | 200–400 |",
        f"| 最大回撤(R) | {oos.get('max_drawdown', 0):.1%} | ≤ 20% |",
        "",
        "## 未达标项",
    ]
    if result.report.failures:
        md.extend(f"- {f}" for f in result.report.failures)
    else:
        md.append("- 无")
    mpath = report_dir / "acceptance_report_v13_triple.md"
    mpath.write_text("\n".join(md), encoding="utf-8")
    return mpath


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--labels", default="data/training/XAUUSD_triple_v3.csv")
    parser.add_argument("--train-balanced", default="data/training/train_balanced_v3.csv")
    parser.add_argument("--out-subdir", default="triple_barrier")
    parser.add_argument("--model-name", default="xgb_triple_v3.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--short-mult", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_v13_triple_training(
        symbol=args.symbol,
        labels_path=args.labels,
        train_balanced_path=args.train_balanced,
        quick=args.quick,
        short_mult=args.short_mult,
        out_subdir=args.out_subdir,
        model_name=args.model_name,
    )
    out_dir = _ROOT / "models" / args.symbol / args.out_subdir
    save_v13_triple_model(result, out_dir, model_name=args.model_name)
    mpath = write_acceptance_report(result, args.symbol)
    logger.info("passed=%s failures=%s", result.report.passed, result.report.failures)
    logger.info("report -> %s", mpath)
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
