#!/usr/bin/env python3
"""启动烛龙多策略引擎（状态机 + AI/趋势/对冲/网格）。"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.engine.multi_strategy_engine import MultiStrategyMt5Runner  # noqa: E402
from zhulong.utils.paths import logs_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="烛龙多策略引擎")
    parser.add_argument(
        "--config",
        default="config/config_multi_strategy.json",
        help="多策略配置文件",
    )
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    args = parser.parse_args()

    log_dir = logs_dir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "multi_strategy.log", encoding="utf-8"),
        ],
    )

    runner = MultiStrategyMt5Runner(args.config, root=_ROOT)

    def _stop(*_):
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)

    runner.start()
    poll = int(runner.config.get("poll_seconds", 30))

    if args.once:
        time.sleep(1.5)
        results = runner.tick_once()
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        runner.stop()
        return 0

    logging.info("多策略引擎已启动 poll=%ds (Ctrl+C 退出)", poll)
    try:
        while True:
            runner.tick_once()
            time.sleep(poll)
    finally:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
