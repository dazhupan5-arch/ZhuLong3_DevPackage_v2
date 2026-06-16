#!/usr/bin/env python3
"""生成 v6 盈亏对齐标签（做多 SL/TP 模拟）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.labels_profit import DEFAULT_MAX_HOLD_BARS, generate_profit_labels
from zhulong.training.lgb.splits import split_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--sl-mult", type=float, default=1.2)
    parser.add_argument("--tp-mult", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    inp = Path(args.input)
    if not inp.is_file():
        alt = _ROOT / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv"
        if not inp.is_file() and alt.is_file():
            inp = alt
    m5 = load_vendor_csv(inp)
    lab = generate_profit_labels(
        m5,
        atr_period=args.atr_period,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        max_hold_bars=args.max_hold_bars,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out)
    print(f"wrote {out}")

    splits = split_indices(lab.index)
    for name, ix in (("all", lab.index), ("train", splits.train), ("val", splits.val), ("test1", splits.test1)):
        sub = lab.loc[ix, "label"]
        print(f"{name}: win={100*(sub==1).mean():.2f}% n={len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
