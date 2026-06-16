#!/usr/bin/env python3
"""预计算每个 M5 bar 的多/空模拟交易质量（R倍数 + 最大不利波动）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from zhulong.analysis.feature_engineering import _adx
from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v13.trade_sim import simulate_trade_trailing

DEFAULT_OUT = "data/features/simulated_trade_quality.parquet"


def precompute_quality(
    m5: pd.DataFrame,
    *,
    max_hold: int = 12,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    trailing: bool = True,
    sample_every: int = 1,
) -> pd.DataFrame:
    atr = _atr_series(m5)
    adx = _adx(m5, 14)
    n = len(m5)
    rows = {
        "sim_r_long": np.full(n, np.nan),
        "sim_mae_long": np.full(n, np.nan),
        "sim_r_short": np.full(n, np.nan),
        "sim_mae_short": np.full(n, np.nan),
        "adx": adx.to_numpy(),
        "atr": atr.to_numpy(),
    }

    close = m5["close"].to_numpy()
    done = 0
    for idx in range(0, n - max_hold - 1, sample_every):
        done += 1
        if done % 50000 == 0:
            print(f"  progress {done}/{n // sample_every} ...")
        a = float(atr.iloc[idx])
        if a <= 0 or np.isnan(a):
            continue
        entry = float(close[idx])
        end = idx + 1 + max_hold
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        if len(hs) == 0:
            continue

        kw = dict(
            max_bars=max_hold,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            trailing=trailing,
        )
        rl = simulate_trade_trailing(1, entry, a, hs, ls, cs, **kw)
        rs = simulate_trade_trailing(-1, entry, a, hs, ls, cs, **kw)
        rows["sim_r_long"][idx] = rl.r_multiple
        rows["sim_mae_long"][idx] = rl.mae_pct
        rows["sim_r_short"][idx] = rs.r_multiple
        rows["sim_mae_short"][idx] = rs.mae_pct

    out = pd.DataFrame(rows, index=m5.index)
    out.index.name = "time"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--features", default="data/training/v13_enhanced/XAUUSD/features.parquet")
    parser.add_argument("--max-hold", type=int, default=12)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--no-trailing", action="store_true")
    parser.add_argument("--all-bars", action="store_true", help="全量 M5（慢）；默认仅特征索引")
    parser.add_argument("--quick", action="store_true", help="每10根采样一次（测试用）")
    args = parser.parse_args()

    m5 = load_vendor_csv(_ROOT / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feat_path = _ROOT / args.features
    if not args.all_bars and feat_path.is_file():
        feat_ix = pd.read_parquet(feat_path).index
        m5 = m5.loc[m5.index.intersection(feat_ix)]
        print(f"Restricted to feature index: {len(m5)} bars")

    sample = 10 if args.quick else 1
    print(f"Computing quality for {len(m5)} bars (sample_every={sample})...")
    q = precompute_quality(
        m5,
        max_hold=args.max_hold,
        sl_mult=args.sl_atr,
        tp_mult=args.tp_atr,
        trailing=not args.no_trailing,
        sample_every=sample,
    )
    out_path = _ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(out_path)
    valid = q["sim_r_long"].notna().sum()
    print(f"Saved {valid} simulated rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
