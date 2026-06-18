"""归因快照字段定义（与 SQLite signals.attribution_json 一致）。"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "v1"

REQUIRED_KEYS = (
    "schema_version",
    "symbol",
    "architecture",
    "bar_time",
    "horizon_direction",
    "horizon_confidence",
    "cognition_direction",
    "cognition_confidence",
    "cognition_regime",
    "rl_raw_action",
    "final_action",
    "filter_reason",
)


def normalize_regime(raw: str | None) -> str:
    r = (raw or "").strip().lower()
    if r in ("trend", "trending", "bull", "bear"):
        return "trend"
    if r in ("ranging", "range", "sideways"):
        return "ranging"
    if r in ("volatile", "volatility", "high_vol"):
        return "volatile"
    return "unknown"


def validate_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    out = dict(snap)
    out["schema_version"] = SCHEMA_VERSION
    out["cognition_regime"] = normalize_regime(str(out.get("cognition_regime", "")))
    for k in REQUIRED_KEYS:
        out.setdefault(k, "")
    return out
