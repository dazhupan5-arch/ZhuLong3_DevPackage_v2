#!/usr/bin/env python3
"""生成训练 smoke 用 M1 CSV（合成数据）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_xauusd_m1.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    n = 8000
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    price = 2300 + np.cumsum(rng.normal(0, 0.15, size=n))
    df = pd.DataFrame(
        {
            "time": idx,
            "open": price,
            "high": price + rng.uniform(0.05, 0.4, n),
            "low": price - rng.uniform(0.05, 0.4, n),
            "close": price + rng.normal(0, 0.05, n),
            "volume": rng.integers(10, 500, n),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"OK: {OUT} rows={len(df)}")


if __name__ == "__main__":
    main()
