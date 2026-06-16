#!/usr/bin/env python3
"""在已训练模型上重新调阈值并输出验收报告（无需重训）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance, get_thresholds
from zhulong.training.lgb.backtest import backtest_signals
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train import (
    _signals_per_day,
    format_threshold_table,
    predict_dual,
    stress_test,
    threshold_sweep,
    tune_threshold,
    tune_threshold_by_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--acceptance-stage", default="v2")
    parser.add_argument("--target-precision", type=float, default=0.40)
    parser.add_argument("--max-signals-per-day", type=float, default=6.0)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    root = _ROOT
    sym = args.symbol
    model_dir = root / "models" / sym / "lgb"
    data_dir = root / "data" / "training" / "lgb" / sym
    m5 = load_vendor_csv(data_dir / f"{sym}_M5.csv")
    feats = pd.read_parquet(data_dir / f"{sym}_features.parquet")
    labels = pd.read_parquet(data_dir / f"{sym}_labels.parquet")
    aligned = feats.join(labels, how="inner")
    splits = split_indices(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    test_ix = splits.test1.intersection(aligned.index)
    stress_ix = splits.stress.intersection(aligned.index)

    meta = joblib.load(model_dir / "lgb_meta.pkl")
    cols = meta["feature_columns"]
    b_long = lgb.Booster(model_file=str(model_dir / "lgb_long.txt"))
    b_short = lgb.Booster(model_file=str(model_dir / "lgb_short.txt"))

    X_va = aligned.loc[va_ix, cols]
    y_va = aligned.loc[va_ix, "label"].values
    y_long = (y_va == 1).astype(int)
    y_short = (y_va == -1).astype(int)
    p_long = b_long.predict(X_va)
    p_short = b_short.predict(X_va)

    print("=== LONG proba stats ===")
    print(f"min={p_long.min():.4f} max={p_long.max():.4f} mean={p_long.mean():.4f} p95={np.percentile(p_long,95):.4f} p99={np.percentile(p_long,99):.4f}")
    thr_long, rows_l = tune_threshold(
        y_long, p_long, va_ix,
        target_precision=args.target_precision,
        max_signals_per_day=args.max_signals_per_day,
    )
    bud_l = tune_threshold_by_budget(y_long, p_long, va_ix, args.max_signals_per_day, args.target_precision)
    print(format_threshold_table(rows_l))
    print(f"best long: thr={thr_long.threshold:.4f} prec={thr_long.precision:.3f} rec={thr_long.recall:.3f} sig/day={thr_long.signals_per_day:.1f}")
    if bud_l:
        print(f"budget long: thr={bud_l.threshold:.4f} prec={bud_l.precision:.3f} rec={bud_l.recall:.3f} sig/day={bud_l.signals_per_day:.1f}")

    print("\n=== SHORT proba stats ===")
    print(f"min={p_short.min():.4f} max={p_short.max():.4f} mean={p_short.mean():.4f} p95={np.percentile(p_short,95):.4f} p99={np.percentile(p_short,99):.4f}")
    thr_short, rows_s = tune_threshold(
        y_short, p_short, va_ix,
        target_precision=args.target_precision,
        max_signals_per_day=args.max_signals_per_day,
    )
    bud_s = tune_threshold_by_budget(y_short, p_short, va_ix, args.max_signals_per_day, args.target_precision)
    print(format_threshold_table(rows_s))
    print(f"best short: thr={thr_short.threshold:.4f} prec={thr_short.precision:.3f} rec={thr_short.recall:.3f} sig/day={thr_short.signals_per_day:.1f}")
    if bud_s:
        print(f"budget short: thr={bud_s.threshold:.4f} prec={bud_s.precision:.3f} rec={bud_s.recall:.3f} sig/day={bud_s.signals_per_day:.1f}")

    # 用 budget 阈值若更优
    use_long = bud_l if bud_l and bud_l.precision >= thr_long.precision else thr_long
    use_short = bud_s if bud_s and bud_s.precision >= thr_short.precision else thr_short

    dirs_va = predict_dual(
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_long.predict(X), b_long.predict(X)])})(),
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_short.predict(X), b_short.predict(X)])})(),
        X_va, use_long.threshold, use_short.threshold,
    )
    y_va_dir = np.zeros(len(y_va), dtype=int)
    y_va_dir[y_va == 1] = 1
    y_va_dir[y_va == -1] = -1
    active = dirs_va != 0
    val_cls = {
        "precision": float((dirs_va[active] == y_va_dir[active]).mean()) if active.any() else 0.0,
        "recall": float(
            (((dirs_va == 1) & (y_va == 1)).sum() + ((dirs_va == -1) & (y_va == -1)).sum())
            / max(((y_va == 1) | (y_va == -1)).sum(), 1)
        ),
        "f1": 0.0,
        "n_signals": int(active.sum()),
        "signals_per_day": _signals_per_day(va_ix, int(active.sum())),
    }
    if val_cls["precision"] + val_cls["recall"] > 0:
        val_cls["f1"] = 2 * val_cls["precision"] * val_cls["recall"] / (val_cls["precision"] + val_cls["recall"] + 1e-9)

    X_test = aligned.loc[test_ix, cols]
    dirs_test = predict_dual(
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_long.predict(X), b_long.predict(X)])})(),
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_short.predict(X), b_short.predict(X)])})(),
        X_test, use_long.threshold, use_short.threshold,
    )
    test1_bt = backtest_signals(m5, test_ix, dirs_test, args.horizon)
    dirs_stress = predict_dual(
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_long.predict(X), b_long.predict(X)])})(),
        type("C", (), {"predict_proba": lambda self, X: np.column_stack([1 - b_short.predict(X), b_short.predict(X)])})(),
        aligned.loc[stress_ix, cols], use_long.threshold, use_short.threshold,
    )
    stress_bt = stress_test(m5, stress_ix, dirs_stress, args.horizon)

    report = evaluate_lgb_acceptance(
        val_cls, test1_bt, stress_bt,
        thresholds=get_thresholds(args.acceptance_stage),
        stage=args.acceptance_stage,
    )
    report.metrics["thresholds"] = {"long": use_long.__dict__, "short": use_short.__dict__}
    out = root / "data" / "training" / "reports" / "lgb" / sym / "retune_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RETUNE ACCEPTANCE ===")
    print(f"passed={report.passed} failures={report.failures}")
    print(f"val={val_cls}")
    print(f"test1={test1_bt}")
    print(f"report -> {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
