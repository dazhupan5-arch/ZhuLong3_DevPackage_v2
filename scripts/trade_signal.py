#!/usr/bin/env python3
"""将 v12 推理结果封装为交易信号并发送绘图指令。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.inference.v12 import V12Inference, V12Signal, load_v12_config  # noqa: E402
from scripts.generate_features import latest_feature_vector  # noqa: E402

logger = logging.getLogger(__name__)


def run_once(
    symbol: str | None = None,
    send_draw: Callable | None = None,
) -> V12Signal:
    """执行一次完整信号生成流程。"""
    cfg = load_v12_config()
    if symbol:
        cfg.symbol = symbol
    row, cols, m5, feats_row = latest_feature_vector(cfg.symbol)
    engine = V12Inference(cfg)
    engine.load()
    sig = engine.build_signal(m5, row, feats_row)
    logger.info(
        "signal %s conf=%.3f reason=%s proba=%s",
        sig.direction, sig.confidence, sig.reject_reason or "ok", sig.probabilities,
    )
    if sig.direction != "flat" and send_draw is not None:
        payload = sig.to_draw_payload(cfg.signal_expiry_minutes)
        if payload and send_draw(payload):
            logger.info("draw_signal sent: %s", sig.signal_id)
        else:
            logger.warning("draw_signal failed: %s", sig.signal_id)
    return sig


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sig = run_once(args.symbol)
    out = {
        "direction": sig.direction,
        "confidence": sig.confidence,
        "entry": sig.entry,
        "sl": sig.sl,
        "tp": sig.tp,
        "signal_id": sig.signal_id,
        "reject_reason": sig.reject_reason,
        "probabilities": sig.probabilities,
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
