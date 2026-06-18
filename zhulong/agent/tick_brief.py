"""V16 统一 tick 数据结构：Structure → Horizon → Execution。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructureSnapshot:
    vector: list[float]
    m5_trend: float = 0.0
    support_dist_atr: float = 0.0
    resistance_dist_atr: float = 0.0
    mtf_align: float = 0.0
    vol_regime: float = 1.0
    zigzag_phase: str = "unknown"


@dataclass
class HorizonForecast:
    horizon_bars: int = 12
    gain_threshold: float = 0.002
    prob_up: float = 0.0
    prob_down: float = 0.0
    prob_flat: float = 1.0
    direction: str = "flat"
    confidence: float = 0.0
    model_id: str = "horizon_v16"

    def to_kn_probs(self) -> list[float]:
        """KN 顺序：0=空 1=平 2=多。"""
        return [self.prob_down, self.prob_flat, self.prob_up]


@dataclass
class ExecutionPlan:
    """V16 统一执行契约：方向 + 入场模式 + 结构锚定价 + SL/TP。"""

    direction: str = "flat"
    action: str = "hold"
    entry_mode: str = "immediate"  # immediate | limit | defer
    entry_target: float = 0.0
    entry_quality: float = 0.0
    size_mult: float = 1.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_reason: str = ""
    block_reason: str = ""
    should_trade: bool = False
    valid_bars: int = 48
    source: str = "composer"
    pos_in_range: float = 0.5
    metadata: dict = field(default_factory=dict)
