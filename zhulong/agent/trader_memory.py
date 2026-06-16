"""交易记忆：胜率 / 盈亏比 / 连亏。"""

from __future__ import annotations

from typing import Any


class TraderMemory:
    def __init__(self, max_len: int = 20) -> None:
        self.trades: list[dict[str, Any]] = []
        self.max_len = max_len

    def add_trade(self, pnl_r: float, exit_time: int | float | str) -> None:
        self.trades.append({"pnl_r": float(pnl_r), "exit_time": exit_time})
        if len(self.trades) > self.max_len:
            self.trades.pop(0)

    def get_winrate(self) -> float:
        if not self.trades:
            return 0.5
        wins = sum(1 for t in self.trades if t["pnl_r"] > 0)
        return wins / len(self.trades)

    def get_avg_rr(self) -> float:
        if not self.trades:
            return 1.0
        wins = [t["pnl_r"] for t in self.trades if t["pnl_r"] > 0]
        losses = [-t["pnl_r"] for t in self.trades if t["pnl_r"] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        return avg_win / max(avg_loss, 1e-9)

    def get_consecutive_losses(self) -> int:
        count = 0
        for t in reversed(self.trades):
            if t["pnl_r"] < 0:
                count += 1
            else:
                break
        return min(count, 10)

    def load_list(self, items: list[dict] | None) -> None:
        self.trades = []
        if not items:
            return
        for it in items[-self.max_len :]:
            self.trades.append({"pnl_r": float(it.get("pnl_r", 0)), "exit_time": it.get("exit_time", 0)})

    def to_list(self) -> list[dict]:
        return list(self.trades)
