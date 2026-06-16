#!/usr/bin/env python3
"""自动调度模拟盘 / 单次 tick 测试。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.engine.scheduler_engine import SchedulerMt5Runner  # noqa: E402

logger = logging.getLogger(__name__)


def run_dry_scheduler_unit() -> dict:
    """不连 MT5，用合成预测验证 SchedulerCore 投票逻辑。"""
    from zhulong.scheduler.context import SchedulerContext
    from zhulong.scheduler.market_state import SchedulerMarketState
    from zhulong.scheduler.scheduler_core import SchedulerCore
    from zhulong.scheduler.types import ModelPrediction
    from zhulong.strategies.base import StrategyContext

    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC")
    n = len(idx)
    m5 = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [10.0] * n,
        },
        index=idx,
    )
    ctx = StrategyContext({"XAUUSD": m5, "USOIL": m5}, config={})
    core = SchedulerCore(
        {
            "weight_allocator": {
                "base_weights": {"XAUUSD": 0.4, "USOIL": 0.6},
                "target_winrate": {"XAUUSD": 0.55, "USOIL": 0.65},
            },
            "state_machine": {"primary_symbol": "XAUUSD", "adx_threshold": 25},
            "risk_manager": {},
            "min_emit_weight": 0.01,
        }
    )
    core.state_machine.state = SchedulerMarketState.TREND
    sched_ctx = SchedulerContext(ctx, core.weight_allocator, core.risk_manager)
    preds = {
        "XAUUSD": ModelPrediction("XAUUSD", 1, 0.82, 100.5, 99.0, 103.0),
        "USOIL": ModelPrediction("USOIL", 1, 0.91, 70.0, 68.0, 74.0),
    }
    outs = core.process_model_outputs(preds, sched_ctx)
    return {
        "dry_run": True,
        "outputs": [
            {
                "symbol": o.symbol,
                "direction": o.direction,
                "confidence": o.confidence,
                "risk_weight": o.risk_weight,
                "weights": o.weights,
            }
            for o in outs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="烛龙自动调度模拟")
    parser.add_argument("--config", default="config/config_scheduler.json")
    parser.add_argument("--dry", action="store_true", help="合成数据单元验证（无需 MT5）")
    parser.add_argument("--once", action="store_true", help="只跑一轮 MT5 tick")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry:
        report = run_dry_scheduler_unit()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    runner = SchedulerMt5Runner(args.config, root=_ROOT)
    runner.start()
    try:
        if args.once:
            import time

            time.sleep(1.5)
            results = runner.tick_once()
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            return 0
        logger.info("调度模拟运行中（Ctrl+C 退出）…")
        import time

        while True:
            runner.tick_once()
            time.sleep(int(runner.config.get("poll_seconds", 30)))
    finally:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
