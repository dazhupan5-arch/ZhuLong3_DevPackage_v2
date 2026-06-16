#!/usr/bin/env python3

"""V13 LightGBM 三分类训练（三重屏障标签，可与 XGBoost 对照）。"""



from __future__ import annotations



import argparse

import json

import logging

import sys

from pathlib import Path



_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



import joblib

import lightgbm as lgb

from sklearn.metrics import classification_report



from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance

from zhulong.training.lgb.backtest import SL_ATR, TP_ATR

from zhulong.training.v10.backtest import backtest_both

from zhulong.training.v11.train import V11_MAX_HOLD

from zhulong.training.v12.train import boost_short_samples

from zhulong.training.v13.train_pipeline import (

    TrainArtifacts,

    load_training_bundle,

)

from zhulong.training.v13.triple import (

    class_sample_weights,

    postprocess_directions,

    tune_precision_thresholds,

)



logger = logging.getLogger(__name__)



V13_LGB_DEFAULT = {

    "objective": "multiclass",

    "num_class": 3,

    "max_depth": 5,

    "learning_rate": 0.03,

    "n_estimators": 1000,

    "subsample": 0.7,

    "colsample_bytree": 0.7,

    "reg_lambda": 3.0,

    "reg_alpha": 1.0,

    "min_child_samples": 20,

    "class_weight": "balanced",

    "n_jobs": -1,

    "random_state": 42,

    "verbose": -1,

}





def _save_lgb_artifacts(

    out_dir: Path,

    artifacts: TrainArtifacts,

    params: dict,

    *,

    model_stem: str = "lgb_triple_v3",

    extra: dict | None = None,

) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts.model.booster_.save_model(str(out_dir / f"{model_stem}.txt"))

    joblib.dump(artifacts.model, out_dir / f"{model_stem}.pkl")

    meta = {

        "model_type": "lgb",

        "params": params,

        "feature_columns": artifacts.feature_columns,

        "long_threshold": artifacts.long_threshold,

        "short_threshold": artifacts.short_threshold,

        "metrics": artifacts.metrics,

        **(extra or {}),

    }

    joblib.dump(meta, out_dir / "params_v13.pkl")

    (out_dir / "feature_columns.json").write_text(

        json.dumps(artifacts.feature_columns, indent=2), encoding="utf-8"

    )

    (out_dir / "config_v13.json").write_text(

        json.dumps(

            {

                "long_threshold": artifacts.long_threshold,

                "short_threshold": artifacts.short_threshold,

                "model_type": "lgb",

                "enhanced": extra.get("enhanced", False) if extra else False,

                "metrics": artifacts.metrics,

            },

            indent=2,

        ),

        encoding="utf-8",

    )





def _write_acceptance_md(report_path: Path, metrics: dict, *, enhanced: bool, n_estimators: int) -> None:

    val = metrics.get("metrics", {}).get("validation", metrics.get("validation", {}))

    test1 = metrics.get("metrics", {}).get("test1", metrics.get("test1", {}))

    passed = metrics.get("passed", False)

    n_trades = test1.get("n_trades", 0)

    days = 365

    daily = n_trades / days if n_trades else 0



    lines = [

        "# V13 Step 1 全量 LightGBM 验收报告",

        "",

        f"> 增强特征: **{'是 (87维)' if enhanced else '否 (68维)'}** | n_estimators={n_estimators} | 早停=50",

        "",

        f"**结论**: {'PASS' if passed else 'PARTIAL/FAIL'}",

        "",

        "## 验证集",

        "| 指标 | 结果 | Step1 目标 |",

        "|------|------|------------|",

        f"| 加权精确率 | {val.get('precision', 0):.1%} | ≥ 50% |",

        f"| 做多精确率 | {val.get('long_precision', 0):.1%} | — |",

        f"| 做空精确率 | {val.get('short_precision', 0):.1%} | — |",

        "",

        "## 2025 样本外",

        "| 指标 | 结果 | 目标 |",

        "|------|------|------|",

        f"| 胜率 | {test1.get('win_rate', 0):.1%} | ≥ 45% |",

        f"| 盈亏比 | {test1.get('avg_rr', 0):.2f} | ≥ 1.5 |",

        f"| 交易笔数 | {n_trades} | 500–750 |",

        f"| 日均交易 | {daily:.2f} | 2–3 |",

        f"| 最大回撤(R) | {test1.get('max_drawdown', 0):.1%} | ≤ 50% |",

        "",

    ]

    failures = metrics.get("failures", [])

    if failures:

        lines.append("## 未达标项")

        for f in failures:

            lines.append(f"- {f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")





def run_lgb_training(

    root: Path,

    symbol: str = "XAUUSD",

    include_enhanced: bool = False,

    quick: bool = False,

    short_mult: int = 1,

    n_estimators: int = 1000,

    early_stopping: int = 50,

    label_type: str = "triple",

    sl_mult: float = 1.2,

    tp_mult: float = 2.0,

    trailing: bool = False,

) -> TrainArtifacts:

    bundle = load_training_bundle(
        root, symbol,
        include_enhanced=include_enhanced or label_type == "quality",
        include_key_levels=label_type == "quality",
        label_type=label_type,
        refresh_features=False,
    )

    cols, m5 = bundle["cols"], bundle["m5"]

    va_ix, te_ix = bundle["va_ix"], bundle["te_ix"]

    aligned = bundle["aligned"]



    train_bal = bundle["train_bal"]

    if short_mult > 1:

        train_bal = boost_short_samples(train_bal, short_mult=short_mult)

    X_tr = train_bal[cols]

    y_tr = train_bal["label"].values.astype(int)

    X_va = aligned.loc[va_ix, cols]

    y_va = aligned.loc[va_ix, "label"].values.astype(int)



    n_est = 150 if quick else n_estimators

    params = {**V13_LGB_DEFAULT, "n_estimators": n_est}

    model = lgb.LGBMClassifier(**params)

    model.fit(

        X_tr,

        y_tr,

        sample_weight=class_sample_weights(y_tr),

        eval_set=[(X_va, y_va)],

        eval_metric="multi_logloss",

        callbacks=[

            lgb.early_stopping(early_stopping, verbose=False),

            lgb.log_evaluation(50 if not quick else 0),

        ],

    )



    clf_rep = classification_report(y_va, model.predict(X_va), target_names=["flat", "long", "short"], zero_division=0)

    logger.info("val clf:\n%s", clf_rep)



    proba_va = model.predict_proba(X_va)

    long_thr, short_thr, sweep, thr_m = tune_precision_thresholds(

        proba_va, y_va, va_ix, m5, target_precision=0.55, max_signals_per_day=6.0

    )

    logger.info(

        "thr=%.2f wprec=%.3f long=%.3f short=%.3f",

        long_thr, thr_m.get("weighted_precision", 0),

        thr_m.get("long_precision", 0), thr_m.get("short_precision", 0),

    )



    proba_te = model.predict_proba(aligned.loc[te_ix, cols])

    dirs_te = postprocess_directions(m5, te_ix, proba_te, long_thr, short_thr)

    oos_bt = backtest_both(
        m5, te_ix, dirs_te,
        max_hold=V11_MAX_HOLD, cooldown_bars=3, max_daily_signals=8,
        sl_mult=sl_mult, tp_mult=tp_mult, trailing=trailing,
    )



    val_metrics = {

        "precision": thr_m.get("weighted_precision", 0.0),

        "recall": thr_m.get("trade_recall", 0.0),

        "long_precision": thr_m.get("long_precision", 0.0),

        "short_precision": thr_m.get("short_precision", 0.0),

        "n_signals": thr_m.get("n_signals", 0),

    }

    report = evaluate_lgb_acceptance(val_metrics, oos_bt, {}, stage="v13_triple")

    report.metrics["threshold_sweep"] = sweep

    report.metrics["label_sl_tp"] = {"sl": sl_mult, "tp": tp_mult, "label_type": label_type}

    report.metrics["model_type"] = "lightgbm"

    report.metrics["enhanced_features"] = include_enhanced

    report.metrics["n_estimators"] = n_est



    return TrainArtifacts(

        model=model,

        model_type="lgb",

        feature_columns=cols,

        long_threshold=long_thr,

        short_threshold=short_thr,

        metrics=report.to_dict(),

    )





def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", default="XAUUSD")

    parser.add_argument("--enhanced", action="store_true", help="使用 87 维增强特征")

    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--refresh-features", action="store_true")

    parser.add_argument("--n-estimators", type=int, default=1000)

    parser.add_argument("--early-stopping", type=int, default=50)

    parser.add_argument("--label-type", choices=["triple", "quality"], default="triple")

    parser.add_argument("--sl-atr", type=float, default=1.0)

    parser.add_argument("--tp-atr", type=float, default=2.0)

    parser.add_argument("--trailing", action="store_true")

    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.refresh_features:

        load_training_bundle(
            _ROOT, args.symbol,
            include_enhanced=args.enhanced or args.label_type == "quality",
            include_key_levels=args.label_type == "quality",
            label_type=args.label_type,
            refresh_features=True,
        )



    result = run_lgb_training(

        _ROOT, args.symbol,

        include_enhanced=args.enhanced,

        quick=args.quick,

        n_estimators=args.n_estimators,

        early_stopping=args.early_stopping,

        label_type=args.label_type,

        sl_mult=args.sl_atr,

        tp_mult=args.tp_atr,

        trailing=args.trailing,

    )



    if args.label_type == "quality":

        out_dir = _ROOT / "models" / args.symbol / "lightgbm"

        model_stem = "lgb_quality_v13"

        report_sub = "quality_v13"

    elif args.enhanced:

        out_dir = _ROOT / "models" / args.symbol / "lightgbm"

        model_stem = "lgb_triple_enhanced"

        report_sub = "step1"

    else:

        out_dir = _ROOT / "models" / args.symbol / "lgb"

        model_stem = "lgb_triple_v3"

        report_sub = "lgb"



    _save_lgb_artifacts(

        out_dir, result, {**V13_LGB_DEFAULT, "n_estimators": args.n_estimators},

        model_stem=model_stem, extra={"enhanced": args.enhanced},

    )



    report_dir = _ROOT / "data" / "training" / "reports" / report_sub / args.symbol

    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "train_report.json").write_text(

        json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8"

    )

    md_name = "acceptance_step1_full.md" if args.enhanced else "acceptance_report.md"

    _write_acceptance_md(

        report_dir / md_name, result.metrics,

        enhanced=args.enhanced, n_estimators=args.n_estimators,

    )



    passed = result.metrics.get("passed", False)

    logger.info("LightGBM saved -> %s/%s.txt", out_dir, model_stem)

    logger.info("LightGBM passed=%s", passed)

    return 0 if passed else 1





if __name__ == "__main__":

    raise SystemExit(main())


