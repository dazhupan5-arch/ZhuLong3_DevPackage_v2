"""适应判断器：监控近期胜率，触发元学习。"""

from __future__ import annotations

from collections import deque
from typing import Any


class AdaptationTrigger:
    def __init__(self, window: int = 20, threshold: float = 0.45) -> None:
        self.window = int(window)
        self.threshold = float(threshold)
        self.recent_wins: deque[bool] = deque(maxlen=self.window)

    def add_result(self, is_win: bool) -> None:
        self.recent_wins.append(bool(is_win))

    def winrate(self) -> float:
        if not self.recent_wins:
            return 0.5
        return sum(self.recent_wins) / len(self.recent_wins)

    def should_adapt(self) -> bool:
        if len(self.recent_wins) < max(5, self.window // 2):
            return False
        return self.winrate() < self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "threshold": self.threshold,
            "winrate": self.winrate(),
            "n": len(self.recent_wins),
        }
