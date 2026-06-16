"""动态权重分配：基础权重 × 滑动胜率 × 置信度。"""

from __future__ import annotations

from typing import Any


class WeightAllocator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.base_weights: dict[str, float] = dict(
            cfg.get("base_weights") or {"XAUUSD": 0.4, "USOIL": 0.6}
        )
        self.winrate_window = int(cfg.get("winrate_window", 50))
        self.target_winrate: dict[str, float] = dict(
            cfg.get("target_winrate") or {"XAUUSD": 0.55, "USOIL": 0.65}
        )
        self.max_winrate_factor = float(cfg.get("max_winrate_factor", 1.2))
        self.history: dict[str, list[bool]] = {s: [] for s in self.base_weights}

    def update(self, symbol: str, is_win: bool) -> None:
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(bool(is_win))
        if len(self.history[symbol]) > self.winrate_window:
            self.history[symbol].pop(0)

    def get_current_winrate(self, symbol: str) -> float:
        hist = self.history.get(symbol) or []
        if not hist:
            return float(self.target_winrate.get(symbol, 0.55))
        return sum(hist) / len(hist)

    def get_recent_winrate(self, symbol: str, window: int = 20) -> float:
        hist = self.history.get(symbol) or []
        if not hist:
            return float(self.target_winrate.get(symbol, 0.55))
        slice_ = hist[-window:]
        return sum(slice_) / len(slice_)

    def compute_weight(self, symbol: str, model_confidence: float) -> float:
        base = float(self.base_weights.get(symbol, 0.5))
        winrate = self.get_current_winrate(symbol)
        target = float(self.target_winrate.get(symbol, 0.55))
        winrate_factor = min(winrate / max(target, 1e-6), self.max_winrate_factor)
        confidence_factor = max(0.0, min(float(model_confidence), 1.0))
        return base * winrate_factor * confidence_factor

    def compute_normalized(self, predictions: dict[str, float]) -> dict[str, float]:
        """predictions: symbol -> raw weight (typically from compute_weight)."""
        total = sum(predictions.values())
        if total <= 1e-9:
            n = max(len(predictions), 1)
            return {s: 1.0 / n for s in predictions}
        return {s: w / total for s, w in predictions.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"history": self.history}

    def load_dict(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        raw = data.get("history") or {}
        for sym, items in raw.items():
            self.history[sym] = [bool(x) for x in items][-self.winrate_window :]
