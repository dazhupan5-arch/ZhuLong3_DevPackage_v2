"""多策略基类与统一信号格式。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class StrategySignal:
    strategy: str
    symbol: str
    direction: str  # buy | sell | flat
    confidence: float
    entry: float
    sl: float
    tp: float
    signal_id: str = ""
    reject_reason: str = ""
    broker_symbol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_id and self.direction != "flat":
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
            self.signal_id = f"multi_{ts}_{self.strategy}_{self.symbol}_{self.direction}"

    def to_draw_payload(self, expiry_minutes: int = 240) -> dict:
        if self.direction == "flat":
            return {}
        sym = self.broker_symbol or self.symbol
        payload = {
            "action": "draw_signal",
            "signal_id": self.signal_id,
            "symbol": sym,
            "direction": self.direction,
            "entry": round(self.entry, 5),
            "sl": round(self.sl, 5),
            "tp": round(self.tp, 5),
            "confidence": round(self.confidence, 4),
            "expiry_minutes": expiry_minutes,
            "strategy": self.strategy,
        }
        if self.metadata:
            payload["meta"] = self.metadata
        return payload


@dataclass
class StrategyContext:
    """策略共享上下文（M5、价格、宏观静默等）。"""

    m5_by_symbol: dict[str, pd.DataFrame]
    config: dict[str, Any]
    macro_silence: bool = False
    bar_time: pd.Timestamp | None = None

    def get_m5(self, symbol: str) -> pd.DataFrame | None:
        return self.m5_by_symbol.get(symbol)

    def get_price(self, symbol: str) -> float:
        m5 = self.get_m5(symbol)
        if m5 is None or m5.empty:
            return 0.0
        return float(m5["close"].iloc[-1])

    def is_macro_silence(self) -> bool:
        return self.macro_silence

    def get_atr(self, symbol: str, period: int = 14) -> float:
        from zhulong.strategies.indicators import atr_series

        m5 = self.get_m5(symbol)
        if m5 is None or len(m5) < period + 2:
            return 0.0
        atr = atr_series(m5, period)
        val = atr.iloc[-1]
        return 0.0 if pd.isna(val) else float(val)

    def get_atr_pct(self, symbol: str, period: int = 14) -> float:
        from zhulong.strategies.indicators import atr_pct

        m5 = self.get_m5(symbol)
        if m5 is None:
            return 0.0
        return atr_pct(m5, period)

    def get_atr_ratio(self, symbol: str, lookback: int = 100) -> float:
        from zhulong.strategies.indicators import atr_ratio

        m5 = self.get_m5(symbol)
        if m5 is None:
            return 1.0
        return atr_ratio(m5, lookback=lookback)

    def get_ema(self, symbol: str, span: int) -> float:
        from zhulong.strategies.indicators import ema

        m5 = self.get_m5(symbol)
        if m5 is None or len(m5) < span:
            return 0.0
        return float(ema(m5["close"], span).iloc[-1])

    def get_adx(self, symbol: str, period: int = 14) -> float:
        from zhulong.strategies.indicators import adx_series

        m5 = self.get_m5(symbol)
        if m5 is None or len(m5) < period * 2:
            return 0.0
        adx = adx_series(m5, period).iloc[-1]
        return 0.0 if pd.isna(adx) else float(adx)


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.state: dict[str, Any] = {}

    @abstractmethod
    def on_bar(self, symbol: str, context: StrategyContext) -> StrategySignal | None:
        pass

    @abstractmethod
    def get_market_condition(self) -> str:
        """trend | volatile | range"""
        pass

    @staticmethod
    def flat(symbol: str, strategy: str, reason: str = "") -> StrategySignal:
        return StrategySignal(
            strategy=strategy,
            symbol=symbol,
            direction="flat",
            confidence=0.0,
            entry=0.0,
            sl=0.0,
            tp=0.0,
            reject_reason=reason,
        )
