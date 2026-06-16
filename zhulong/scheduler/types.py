"""调度器数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelPrediction:
    symbol: str
    direction: int  # 1 long, -1 short, 0 flat
    confidence: float
    entry: float
    sl: float
    tp: float
    signal_id: str = ""
    broker_symbol: str = ""
    reject_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_direction_str(
        cls,
        symbol: str,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        **kwargs: Any,
    ) -> ModelPrediction:
        d = 1 if direction == "buy" else (-1 if direction == "sell" else 0)
        return cls(symbol, d, confidence, entry, sl, tp, **kwargs)


@dataclass
class SchedulerOutput:
    """调度器最终输出（可映射为 StrategySignal）。"""

    symbol: str
    direction: str
    confidence: float
    entry: float
    sl: float
    tp: float
    signal_id: str
    strategy: str = "scheduler_ai"
    broker_symbol: str = ""
    risk_weight: float = 0.0
    market_state: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    reject_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
