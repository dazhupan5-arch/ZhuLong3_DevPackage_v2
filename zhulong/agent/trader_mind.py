"""L3 交易员层：委托 ExecutionComposer 产出执行计划。"""

from __future__ import annotations

from typing import Any

from zhulong.agent.execution_composer import ExecutionComposer
from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot


class TraderMind(ExecutionComposer):
    """向后兼容别名：TraderMind.plan → ExecutionComposer.compose。"""

    def plan(
        self,
        forecast: HorizonForecast,
        snapshot: StructureSnapshot,
        *,
        close: float,
        atr: float,
        consecutive_losses: int = 0,
        regime: str = "",
        pos_in_range: float = 0.5,
        kn2_dec: dict[str, Any] | None = None,
        horizon_flat: bool = False,
    ) -> ExecutionPlan:
        return self.compose(
            forecast,
            snapshot,
            close=close,
            atr=atr,
            pos_in_range=pos_in_range,
            kn2_dec=kn2_dec,
            consecutive_losses=consecutive_losses,
            regime=regime,
            horizon_flat=horizon_flat,
        )
