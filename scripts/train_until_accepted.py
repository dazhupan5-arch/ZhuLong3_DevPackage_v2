#!/usr/bin/env python3
"""按阶段性验收标准循环训练，直至全部指标通过。"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.acceptance import LabelParams, LABEL_MODE
from zhulong.training.pipeline import (
    FEATURE_MODE,
    build_feature_matrix,
    load_m5_csv,
    plot_oos_equity,
    save_model,
    save_rejected,
    train_one,
)
from zhulong.utils.train_logging import flush_train_logs, setup_train_logging

logger = logging.getLogger(__name__)
REPORTS = _ROOT / "data" / "training" / "reports"
STATE_PATH = _ROOT / "data" / "training" / "train_state.json"
PID_PATH = _ROOT / "data" / "training" / "train.pid"


def _write_pid() -> None:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(__import__("os").getpid()), encoding="utf-8")


def _save_state(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    flush_train_logs()


def _grid() -> list[dict]:
    base = {
        "horizon": [8, 12, 16],
        "return_threshold_pct": [0.12, 0.15, 0.18],
        "xgb_depth": [3, 4],
        "xgb_trees": [100, 150, 200],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "prob_threshold": [0.55, 0.60, 0.65, 0.70],
        "long_weight": [0.8, 1.0, 1.2],
        "short_weight": [0.8, 1.0, 1.2],
    }
    keys = list(base.keys())
    all_combos = [dict(zip(keys, c)) for c in itertools.product(*[base[k] for k in keys])]
    random.shuffle(all_combos)
    return all_combos


def _score(report) -> float:
    v = report.metrics["validation"]
    bt = report.metrics.get("backtest_validation", {})
    prec, rec, f1 = v["precision"], v["recall"], v["f1"]
    rr = bt.get("avg_rr", 0.0)
    gap = report.metrics.get("train_val_precision_gap", 0.5)
    return (
        prec * 0.40
        + min(rec / 0.15, 1.0) * 0.25
        + min(f1 / 0.25, 1.0) * 0.15
        + min(rr / 1.3, 1.0) * 0.10
        + max(0.0, 1.0 - gap / 0.25) * 0.10
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="烛龙验收训练循环（直到达标）")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--m5-csv", default="data/training/XAUUSD_M5_dense.csv")
    parser.add_argument("--max-runs", type=int, default=0, help="0 = 无限循环直到达标")
    parser.add_argument("--log-file", default="data/training/train_daemon.log")
    args = parser.parse_args()

    log_path = setup_train_logging(_ROOT / args.log_file)
    _write_pid()
    logger.info("train_until_accepted started pid=%s log=%s", PID_PATH.read_text(), log_path)

    csv_path = _ROOT / args.m5_csv if not Path(args.m5_csv).is_absolute() else Path(args.m5_csv)
    if not csv_path.is_file():
        logger.error("Missing CSV: %s", csv_path)
        return 1

    try:
        m5 = load_m5_csv(csv_path)
        logger.info("M5 rows=%s range=%s .. %s", len(m5), m5.index.min(), m5.index.max())
        logger.info("Labels=%s features=%s", LABEL_MODE, FEATURE_MODE)

        build_feature_matrix(m5)
        logger.info("Feature matrix ready")

        combos = _grid()
        best_score = -1.0
        best_result = None
        run_i = 0
        round_no = 1

        while True:
            logger.info("===== round %s =====", round_no)
            for cfg in combos:
                run_i += 1
                if args.max_runs and run_i > args.max_runs:
                    logger.error("Reached max-runs=%s best_score=%.3f", args.max_runs, best_score)
                    return 1

                lp = LabelParams(
                    horizon=int(cfg["horizon"]),
                    return_threshold_pct=float(cfg["return_threshold_pct"]),
                )
                logger.info("run %s cfg=%s", run_i, cfg)
                t0 = time.time()
                try:
                    result = train_one(
                        m5,
                        lp,
                        xgb_depth=int(cfg["xgb_depth"]),
                        xgb_trees=int(cfg["xgb_trees"]),
                        long_weight=float(cfg["long_weight"]),
                        short_weight=float(cfg["short_weight"]),
                        prob_threshold=float(cfg["prob_threshold"]),
                        subsample=float(cfg["subsample"]),
                        colsample_bytree=float(cfg["colsample_bytree"]),
                    )
                except Exception:
                    logger.error("run %s failed:\n%s", run_i, traceback.format_exc())
                    flush_train_logs()
                    continue

                if result is None:
                    continue

                sc = _score(result.report)
                v = result.report.metrics["validation"]
                _save_state({
                    "run": run_i,
                    "round": round_no,
                    "cfg": cfg,
                    "best_score": max(best_score, sc),
                    "last": {"prec": v["precision"], "recall": v["recall"], "f1": v["f1"]},
                    "failures": result.report.failures,
                })

                if sc > best_score:
                    best_score = sc
                    best_result = result
                    logger.info(
                        "new best score=%.3f prec=%.3f rec=%.3f f1=%.3f gap=%.3f (%.0fs)",
                        sc,
                        v["precision"],
                        v["recall"],
                        v["f1"],
                        result.report.metrics.get("train_val_precision_gap", 0),
                        time.time() - t0,
                    )

                if result.passed:
                    out = save_model(result, args.symbol, REPORTS)
                    plot_oos_equity(result, REPORTS / args.symbol / "oos_equity.png")
                    bin_dst = (
                        _ROOT / "src" / "ZhuLong.App" / "bin" / "x64" / "Release"
                        / "net8.0-windows10.0.19041.0" / "win-x64" / "models" / args.symbol
                    )
                    if bin_dst.parent.is_dir():
                        import shutil
                        if bin_dst.exists():
                            shutil.rmtree(bin_dst)
                        shutil.copytree(out, bin_dst)
                    logger.info("ACCEPTANCE PASSED. Model published: %s", out)
                    flush_train_logs()
                    return 0

                if run_i % 10 == 0 and best_result:
                    save_rejected(best_result, args.symbol, REPORTS)

            round_no += 1
            random.shuffle(combos)
            logger.info("round %s done best_score=%.3f, continue...", round_no - 1, best_score)
            flush_train_logs()
            time.sleep(2)

    except Exception:
        logger.critical("fatal error:\n%s", traceback.format_exc())
        flush_train_logs()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
