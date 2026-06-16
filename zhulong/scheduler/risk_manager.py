"""调度层风控：总回撤 / 日损 / 连亏（R 单位）。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


class SchedulerRiskManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.max_total_drawdown_r = float(cfg.get("max_total_drawdown_r", 0.3))
        self.max_daily_loss_r = float(cfg.get("max_daily_loss_r", 0.15))
        self.max_consecutive_losses = int(cfg.get("max_consecutive_losses", 5))
        self.total_pnl_r = float(cfg.get("initial_total_pnl_r", 0.0))
        self.daily_pnl_r = 0.0
        self.consecutive_losses = int(cfg.get("initial_consecutive_losses", 0))
        self._last_date: date | None = None
        self.block_reason = ""

    def _roll_date(self, ts: datetime) -> None:
        d = ts.date()
        if self._last_date != d:
            self.daily_pnl_r = 0.0
            self._last_date = d

    def update(self, pnl_r: float, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.utcnow()
        self._roll_date(ts)
        self.total_pnl_r += float(pnl_r)
        self.daily_pnl_r += float(pnl_r)
        if pnl_r < 0:
            self.consecutive_losses += 1
        elif pnl_r > 0:
            self.consecutive_losses = 0

    def can_open_position(self) -> bool:
        self.block_reason = ""
        if self.total_pnl_r < -self.max_total_drawdown_r:
            self.block_reason = f"总回撤 {self.total_pnl_r:.2f}R 超限"
            return False
        if self.daily_pnl_r < -self.max_daily_loss_r:
            self.block_reason = f"日损 {self.daily_pnl_r:.2f}R 超限"
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.block_reason = f"连亏 {self.consecutive_losses} 次"
            return False
        return True

    def reset_daily(self) -> None:
        self.daily_pnl_r = 0.0

    def status(self) -> dict[str, Any]:
        return {
            "total_pnl_r": round(self.total_pnl_r, 4),
            "daily_pnl_r": round(self.daily_pnl_r, 4),
            "consecutive_losses": self.consecutive_losses,
            "can_open": self.can_open_position(),
            "block_reason": self.block_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pnl_r": self.total_pnl_r,
            "daily_pnl_r": self.daily_pnl_r,
            "consecutive_losses": self.consecutive_losses,
            "last_date": self._last_date.isoformat() if self._last_date else None,
        }

    def load_dict(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        self.total_pnl_r = float(data.get("total_pnl_r", self.total_pnl_r))
        self.daily_pnl_r = float(data.get("daily_pnl_r", 0.0))
        self.consecutive_losses = int(data.get("consecutive_losses", 0))
        ld = data.get("last_date")
        if ld:
            try:
                self._last_date = date.fromisoformat(str(ld))
            except ValueError:
                self._last_date = None
