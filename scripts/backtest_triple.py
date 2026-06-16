#!/usr/bin/env python3
"""v11/v13 三分类样本外回测：ADX过滤、移动止损、元标签、自定义阈值。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd
import xgboost as xgb

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v10.backtest import backtest_both
from zhulong.training.v11.train import V11_COOLDOWN, V11_MAX_DAILY, V11_MAX_HOLD, proba_to_directions
from zhulong.training.v13.backtest_runner import run_v13_backtest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", choices=["val", "test1", "test2025"], default="test2025")
    parser.add_argument("--version", choices=["v11", "v13_triple", "v3"], default="v3")
    parser.add_argument("--model", default="")
    parser.add_argument("--threshold", type=float, default=None, help="统一阈值（覆盖 long/short）")
    parser.add_argument("--long-thr", type=float, default=0.55)
    parser.add_argument("--short-thr", type=float, default=0.45)
    parser.add_argument("--meta", action="store_true")
    parser.add_argument("--meta-model", default="")
    parser.add_argument("--meta-threshold", type=float, default=0.55)
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--stop-loss", action="store_true", help="启用动态 SL/TP（默认 sl=1.0 tp=2.0）")
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--time-stop-bars", type=int, default=12)
    parser.add_argument("--trailing", action="store_true", help="移动止损：1.0ATR保本，1.5ATR收紧")
    parser.add_argument("--adx-filter", action="store_true", help="ADX(14)>25 才允许开仓")
    parser.add_argument("--adx-min", type=float, default=25.0)
    args = parser.parse_args()

    root = _ROOT
    sl_mult = args.sl_atr
    tp_mult = args.tp_atr

    if args.version in ("v13_triple", "v3"):
        meta_path = args.meta_model
        if args.meta and not meta_path:
            for candidate in (
                "models/XAUUSD/meta_lgb/meta_lgb_model.pkl",
                "models/XAUUSD/meta_label/meta_label.pkl",
            ):
                if (root / candidate).is_file():
                    meta_path = candidate
                    break

        thr = args.threshold
        long_thr = thr if thr is not None else args.long_thr
        short_thr = thr if thr is not None else args.short_thr

        report = run_v13_backtest(
            root,
            symbol=args.symbol,
            split=args.split,
            model_path=args.model or None,
            long_thr=long_thr,
            short_thr=short_thr,
            enhanced=args.enhanced if args.enhanced else None,
            meta_model_path=meta_path or None,
            meta_threshold=args.meta_threshold,
            use_meta=args.meta,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            max_hold=args.time_stop_bars,
            trailing=args.trailing or args.stop_loss,
            adx_filter=args.adx_filter,
            adx_min=args.adx_min,
        )

        out_dir = root / "data" / "training" / "reports" / "v13_triple" / args.symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_meta" if args.meta else ""
        path = out_dir / f"backtest_{args.split}_v13_triple{suffix}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        bt_base = report["backtest_baseline"]
        print(f"=== v13_triple backtest {args.split} ({report.get('model_type', '?')}) ===")
        print(f"long={report['long_thr']} short={report['short_thr']} SL={sl_mult} TP={tp_mult} "
              f"trailing={report.get('trailing')} adx>{report.get('adx_min') if report.get('adx_filter') else 'off'}")
        print("--- 基线 ---")
        for k, v in bt_base.items():
            print(f"  {k}: {v}")

        if args.meta and "backtest_meta" in report:
            print(f"--- 元标签 (th={args.meta_threshold}) ---")
            for k, v in report["backtest_meta"].items():
                print(f"  {k}: {v}")
            c = report["comparison"]
            print(f"  胜率: {c['win_rate_before']:.1%} -> {c['win_rate_after']:.1%}")
            print(f"  笔数: {c['n_trades_before']} -> {c['n_trades_after']}")
            print(f"  回撤: {c['max_drawdown_before']:.1%} -> {c['max_drawdown_after']:.1%}")

        print(f"report -> {path}")
        return 0

    meta = joblib.load(root / "models" / args.symbol / "v11" / "v11_meta.pkl")
    cfg = json.loads((root / "models" / args.symbol / "v11" / "config_v11.json").read_text(encoding="utf-8"))
    cols = meta["feature_columns"]
    long_thr = cfg.get("long_threshold", meta["long_threshold"])
    short_thr = cfg.get("short_threshold", meta["short_threshold"])

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(root / "data" / "training" / "v8" / args.symbol / "features.parquet")
    splits = split_indices(feats.index)
    ix = getattr(splits, args.split).intersection(feats.index)

    model = xgb.XGBClassifier()
    model.load_model(str(root / "models" / args.symbol / "v11" / "xgb_triple.json"))
    proba = model.predict_proba(feats.loc[ix, cols])
    dirs = proba_to_directions(proba, long_thr, short_thr)
    bt = backtest_both(
        m5, ix, dirs,
        max_hold=args.time_stop_bars, cooldown_bars=V11_COOLDOWN, max_daily_signals=V11_MAX_DAILY,
        sl_mult=sl_mult, tp_mult=tp_mult, trailing=args.trailing,
    )

    report = {"version": "v11", "split": args.split, "backtest": bt}
    out_dir = root / "data" / "training" / "reports" / "v11" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_{args.split}_v11.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"=== v11 backtest {args.split} ===")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
