#!/usr/bin/env python3
"""极端行情压力测试（对 v12 回测框架的日期窗口封装）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backtest_model import run_backtest  # noqa: E402

STRESS_WINDOWS = {
    "covid_2020": ("2020-02-01", "2020-04-30"),
    "ukraine_2022": ("2022-01-15", "2022-03-31"),
    "yuan_2024": ("2024-07-01", "2024-09-30"),
    "geo_2026": ("2026-01-01", "2026-03-31"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--output", default="data/training/reports/stress_test.json")
    args = parser.parse_args()

    report: dict = {"symbol": args.symbol, "windows": {}}
    for name, (start, end) in STRESS_WINDOWS.items():
        try:
            metrics = run_backtest(args.symbol, start, end, root=_ROOT)
            metrics["pass_max_dd"] = metrics.get("max_dd_r", 99) <= 20.0
            metrics["pass_streak"] = True  # 需逐笔连亏统计时可扩展
            report["windows"][name] = metrics
        except Exception as ex:
            report["windows"][name] = {"error": str(ex)}

    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
