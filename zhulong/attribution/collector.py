"""从 TradingAgent tick 输出构建归因快照。"""

from __future__ import annotations

from typing import Any

from zhulong.attribution.schema import SCHEMA_VERSION, normalize_regime


def build_attribution_snapshot(
    *,
    symbol: str,
    bar_time: str,
    architecture: str,
    horizon_direction: str = "",
    horizon_confidence: float = 0.0,
    horizon_min_confidence: float = 0.0,
    cognition_direction: str = "",
    cognition_confidence: float = 0.0,
    cognition_regime: str = "",
    cognition_regime_confidence: float = 0.0,
    rl_raw_action: str = "",
    final_action: str = "",
    filter_reason: str = "",
    kn2_should_trade: bool = False,
    kn2_action: str = "",
    kn2_confidence: float = 0.0,
    kn2_shadow_mode: bool = False,
    pos_in_range: float = 0.5,
    structure_location_gate: bool = True,
    causal_pred: float = 0.0,
    signal_direction: str = "",
    block_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol.upper(),
        "bar_time": bar_time,
        "architecture": architecture or "legacy",
        "horizon_direction": horizon_direction or "flat",
        "horizon_confidence": round(float(horizon_confidence), 4),
        "horizon_min_confidence": round(float(horizon_min_confidence), 4),
        "cognition_direction": cognition_direction or "flat",
        "cognition_confidence": round(float(cognition_confidence), 4),
        "cognition_regime": normalize_regime(cognition_regime),
        "cognition_regime_confidence": round(float(cognition_regime_confidence), 4),
        "rl_raw_action": rl_raw_action or "hold",
        "final_action": final_action or "hold",
        "filter_reason": filter_reason or "",
        "kn2_should_trade": bool(kn2_should_trade),
        "kn2_action": kn2_action or "",
        "kn2_confidence": round(float(kn2_confidence), 4),
        "kn2_shadow_mode": bool(kn2_shadow_mode),
        "pos_in_range": round(float(pos_in_range), 4),
        "structure_location_gate": bool(structure_location_gate),
        "causal_pred": round(float(causal_pred), 6),
        "signal_direction": signal_direction or "flat",
        "block_reason": block_reason or "",
    }
    if extra:
        snap.update(extra)
    return snap


def snapshot_from_tick_info(info: dict[str, Any], *, cognition: dict[str, Any] | None = None) -> dict[str, Any]:
    cog = cognition or info.get("cognition") or {}
    return build_attribution_snapshot(
        symbol=str(info.get("symbol", "")),
        bar_time=str(info.get("bar_time", "")),
        architecture=str(info.get("architecture", "legacy")),
        horizon_direction=str(info.get("horizon_direction", "")),
        horizon_confidence=float(info.get("horizon_confidence", 0)),
        horizon_min_confidence=float(info.get("horizon_min_confidence", 0)),
        cognition_direction=str(info.get("cognition_direction", cog.get("regime", ""))),
        cognition_confidence=float(info.get("cognition_confidence", cog.get("confidence", 0))),
        cognition_regime=str(info.get("cognition_regime", cog.get("regime", ""))),
        cognition_regime_confidence=float(info.get("cognition_regime_confidence", cog.get("regime_confidence", 0))),
        rl_raw_action=str(info.get("rl_raw_action", "")),
        final_action=str(info.get("action", "")),
        filter_reason=str(info.get("filter_reason", "")),
        kn2_should_trade=bool(info.get("kn2_should_trade", False)),
        kn2_action=str(info.get("kn2_action", "")),
        kn2_confidence=float(info.get("kn2_confidence", 0)),
        kn2_shadow_mode=bool(info.get("kn2_shadow_mode", False)),
        pos_in_range=float(info.get("pos_in_range", 0.5)),
        causal_pred=float(info.get("causal_pred", 0)),
        signal_direction=str((info.get("signal") or {}).get("direction", "flat")),
    )
