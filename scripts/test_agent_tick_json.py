#!/usr/bin/env python3
"""回归：agent_tick 响应必须可被 strict JSON 解析（无 NaN/Inf）。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhulong.utils.json_safe import dumps_strict, json_safe  # noqa: E402


def _assert_strict_parse(payload: dict) -> None:
    text = dumps_strict(payload)
    if "NaN" in text or "Infinity" in text:
        raise AssertionError(f"serialized text contains non-finite token: {text[:200]}")
    parsed = json.loads(text)
    json.dumps(parsed, allow_nan=False)


def test_json_safe_scalars() -> None:
    bad = {"a": float("nan"), "b": float("inf"), "c": 1.0}
    safe = json_safe(bad)
    assert safe["a"] is None
    assert safe["b"] is None
    assert safe["c"] == 1.0
    _assert_strict_parse(safe)


def test_tick_like_payload() -> None:
    import numpy as np

    payload = {
        "ok": True,
        "agent": True,
        "results": [
            {
                "symbol": "XAUUSD",
                "horizon_confidence": np.float64(0.521),
                "cognition_confidence": float("nan"),
                "signal": {
                    "confidence": np.float32(0.48),
                    "metadata": {"knowledge_probs": [0.1, np.nan, 0.2]},
                },
            }
        ],
    }
    safe = json_safe(payload)
    _assert_strict_parse(safe)
    assert safe["results"][0]["cognition_confidence"] is None


def test_duplicate_bar_skipped_payload() -> None:
    payload = {
        "ok": True,
        "agent": True,
        "results": [
            {
                "symbol": "XAUUSD",
                "skipped": True,
                "reason": "duplicate_bar",
                "signal": {"direction": "flat", "reject_reason": "duplicate_bar"},
            }
        ],
        "skipped_only": True,
    }
    _assert_strict_parse(json_safe(payload))


def test_duplicate_bar_skipped() -> None:
    payload = {
        "ok": True,
        "agent": True,
        "results": [
            {
                "symbol": "XAUUSD",
                "skipped": True,
                "reason": "duplicate_bar",
                "signal": {"direction": "flat", "reject_reason": "duplicate_bar"},
            }
        ],
        "skipped_only": True,
    }
    _assert_strict_parse(payload)


def main() -> int:
    test_json_safe_scalars()
    test_tick_like_payload()
    test_duplicate_bar_skipped()
    test_duplicate_bar_skipped_payload()
    print("test_agent_tick_json: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
