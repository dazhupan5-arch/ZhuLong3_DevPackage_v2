#!/usr/bin/env python3
"""Quick v12 live inference probe (MT5 required for predict)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from zhulong.inference_engine import InferenceEngine


def main() -> int:
    eng = InferenceEngine({})
    if not eng.validate_symbol_models("XAUUSD"):
        print("FAIL validate XAUUSD")
        return 1
    eng.load("XAUUSD")
    try:
        r = eng.predict("XAUUSD", np.zeros((60, 30), dtype=np.float32), np.zeros(10), np.zeros(8))
        print("OK v12 live probe:", r)
    except Exception as ex:
        print("WARN predict:", ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
