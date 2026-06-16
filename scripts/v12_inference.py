#!/usr/bin/env python3
"""v12 模型加载 + 推理 + 后处理（CLI）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.inference.v12 import V12Inference, load_v12_config  # noqa: E402
from scripts.generate_features import latest_feature_vector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="v12 单次推理")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--config", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_v12_config(args.config or None)
    cfg.symbol = args.symbol

    row, cols, m5, feats_row = latest_feature_vector(cfg.symbol)
    engine = V12Inference(cfg)
    engine.load()
    proba = engine.predict_proba(row)
    direction, conf = engine.infer_direction(proba, m5, feats_row, m5.index[-1])
    sig = engine.build_signal(m5, row, feats_row)

    result = {
        "bar_time": str(m5.index[-1]),
        "close": float(m5["close"].iloc[-1]),
        "probabilities": {"flat": float(proba[0]), "long": float(proba[1]), "short": float(proba[2])},
        "direction_int": direction,
        "confidence": conf,
        "signal": {
            "direction": sig.direction,
            "entry": sig.entry,
            "sl": sig.sl,
            "tp": sig.tp,
            "signal_id": sig.signal_id,
            "reject_reason": sig.reject_reason,
        },
        "thresholds": {
            "long": cfg.long_threshold,
            "short": cfg.short_threshold,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
