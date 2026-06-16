"""智能体调度：元学习 + 适应触发（Phase 6）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zhulong.agent.adaptation_trigger import AdaptationTrigger
from zhulong.agent.meta_learner import MetaLearner
from zhulong.utils.paths import resolve_writable_data_path

logger = logging.getLogger(__name__)


class AgentScheduler:
    """协调实盘 tick 后的元学习与适应判断。"""

    def __init__(self, config: dict[str, Any], root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.config = config
        ml_cfg = config.get("meta_learning") or {}
        at_cfg = config.get("adaptation_trigger") or {}

        state_dir = resolve_writable_data_path(ml_cfg.get("state_dir", "data/meta_learning"))

        self.meta = MetaLearner(ml_cfg, state_dir=state_dir)
        self.trigger = AdaptationTrigger(
            window=int(at_cfg.get("window", 20)),
            threshold=float(at_cfg.get("threshold", 0.45)),
        )
        self._tick_steps = 0

    def attach_policy(self, policy: Any) -> None:
        self.meta.attach_policy(policy)

    def on_trade_closed(self, trajectory: list[dict], pnl_r: float) -> dict[str, Any]:
        is_win = pnl_r > 0
        self.trigger.add_result(is_win)
        self.meta.add_trajectory(trajectory)
        result: dict[str, Any] = {"is_win": is_win, "pnl_r": pnl_r}

        if self.trigger.should_adapt():
            logger.info("适应触发：近%d笔胜率 %.1f%% < 阈值", len(self.trigger.recent_wins), self.trigger.winrate() * 100)
            result["meta_update"] = self.meta.meta_update()
            result["adaptation_triggered"] = True
        return result

    def on_tick(self) -> dict[str, Any] | None:
        self._tick_steps += 1
        self.meta.record_step()
        if self.meta.should_run_scheduled_update():
            return self.meta.meta_update()
        return None

    def apply_action_bias(self, action: int, n_actions: int = 6) -> int:
        """用元学习偏置微调离散动作（仅当偏置显著时）。"""
        bias = self.meta.action_bias()
        if bias.size < n_actions or float(bias.max() - bias.min()) < 1e-4:
            return action
        adjusted = int(action)
        if action == 0 and bias[1] > bias[3] + 0.05:
            adjusted = 1
        elif action == 0 and bias[3] > bias[1] + 0.05:
            adjusted = 3
        return adjusted
