"""实机信号通用类型（冷却状态 / 绘图 payload）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LiveSignal:
    direction: str  # buy | sell | flat
    confidence: float
    entry: float
    sl: float
    tp: float
    signal_id: str
    symbol: str
    probabilities: list[float]
    reject_reason: str = ""

    def to_draw_payload(self, expiry_minutes: int = 240) -> dict:
        if self.direction == "flat":
            return {}
        return {
            "action": "draw_signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": round(self.confidence, 4),
            "expiry_minutes": expiry_minutes,
        }


# 兼容旧模块名
V12Signal = LiveSignal


@dataclass
class CooldownState:
    last_long_utc: str | None = None
    last_short_utc: str | None = None
    last_m5_bar: str | None = None
    daily_counts: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "last_long_utc": self.last_long_utc,
                    "last_short_utc": self.last_short_utc,
                    "last_m5_bar": self.last_m5_bar,
                    "daily_counts": self.daily_counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CooldownState:
        if not path.is_file():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_long_utc=d.get("last_long_utc"),
                last_short_utc=d.get("last_short_utc"),
                last_m5_bar=d.get("last_m5_bar"),
                daily_counts={str(k): int(v) for k, v in (d.get("daily_counts") or {}).items()},
            )
        except Exception as ex:
            logger.warning("状态文件损坏，重置: %s", ex)
            return cls()


from zhulong.utils.paths import resolve_runtime_path


def resolve_path(root: Path, p: str) -> Path:
    return resolve_runtime_path(p, root=root)
