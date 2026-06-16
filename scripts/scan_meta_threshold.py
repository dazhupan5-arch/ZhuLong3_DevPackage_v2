#!/usr/bin/env python3
"""扫描元标签阈值（支持 ADX 过滤 + 移动止损）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from zhulong.training.v13.backtest_runner import run_v13_backtest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta-model", required=True)
    parser.add_argument("--split", default="test2025")
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--long-thr", type=float, default=0.55)
    parser.add_argument("--short-thr", type=float, default=0.45)
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--output", default="data/training/reports/meta_scan/meta_threshold_scan.csv")
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--time-stop-bars", type=int, default=12)
    parser.add_argument("--stop-loss", action="store_true")
    parser.add_argument("--trailing", action="store_true")
    parser.add_argument("--adx-filter", action="store_true")
    parser.add_argument("--adx-min", type=float, default=25.0)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    trailing = args.trailing or args.stop_loss

    baseline = run_v13_backtest(
        _ROOT,
        symbol=args.symbol,
        split=args.split,
        model_path=args.model,
        long_thr=args.long_thr,
        short_thr=args.short_thr,
        enhanced=args.enhanced,
        use_meta=False,
        sl_mult=args.sl_atr,
        tp_mult=args.tp_atr,
        max_hold=args.time_stop_bars,
        trailing=trailing,
        adx_filter=args.adx_filter,
        adx_min=args.adx_min,
    )
    bt0 = baseline["backtest_baseline"]
    rows = [{
        "threshold": "baseline",
        "win_rate": bt0.get("win_rate", 0),
        "n_trades": bt0.get("n_trades", 0),
        "avg_rr": bt0.get("avg_rr", 0),
        "max_drawdown": bt0.get("max_drawdown", 0),
        "total_pnl_r": bt0.get("total_pnl_r", 0),
        "filter_rate": 0.0,
    }]

    for th in thresholds:
        result = run_v13_backtest(
            _ROOT,
            symbol=args.symbol,
            split=args.split,
            model_path=args.model,
            long_thr=args.long_thr,
            short_thr=args.short_thr,
            enhanced=args.enhanced,
            meta_model_path=args.meta_model,
            meta_threshold=th,
            use_meta=True,
            sl_mult=args.sl_atr,
            tp_mult=args.tp_atr,
            max_hold=args.time_stop_bars,
            trailing=trailing,
            adx_filter=args.adx_filter,
            adx_min=args.adx_min,
        )
        bt = result["backtest_meta"]
        mf = result.get("meta_filter", {})
        rows.append({
            "threshold": th,
            "win_rate": bt.get("win_rate", 0),
            "n_trades": bt.get("n_trades", 0),
            "avg_rr": bt.get("avg_rr", 0),
            "max_drawdown": bt.get("max_drawdown", 0),
            "total_pnl_r": bt.get("total_pnl_r", 0),
            "filter_rate": mf.get("filter_rate", 0),
        })
        print(
            f"th={th:.2f} win={bt.get('win_rate', 0):.1%} trades={bt.get('n_trades', 0)} "
            f"dd={bt.get('max_drawdown', 0):.1%} filter={mf.get('filter_rate', 0):.1%}"
        )

    df = pd.DataFrame(rows)
    out = _ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False)}\nSaved -> {out}")

    viable = df[(df["threshold"] != "baseline") & (df["win_rate"] >= 0.5)]
    if not viable.empty:
        best = viable.loc[viable["max_drawdown"].idxmin()]
        print(f"Best (win>=50%): th={best['threshold']} win={best['win_rate']:.1%} dd={best['max_drawdown']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
