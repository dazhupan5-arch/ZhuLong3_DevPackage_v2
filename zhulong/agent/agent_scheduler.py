"""智能体调度：元学习 + 适应触发 + 归因联动（Phase 6）。"""

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
        ml_cfg = self._resolve_meta_cfg(config)
        at_cfg = config.get("adaptation_trigger") or {}

        state_dir = resolve_writable_data_path(ml_cfg.get("state_dir", "data/meta_learning"))

        self.meta = MetaLearner(ml_cfg, state_dir=state_dir)
        self.trigger = AdaptationTrigger(
            window=int(at_cfg.get("window", 20)),
            threshold=float(at_cfg.get("threshold", 0.45)),
        )
        self._tick_steps = 0
        self._last_regime = "unknown"

    @staticmethod
    def _resolve_meta_cfg(config: dict[str, Any]) -> dict[str, Any]:
        ml_cfg = dict(config.get("meta_learning") or {})
        te_ml = (config.get("trading_env") or {}).get("meta_learning") or {}
        if te_ml.get("enabled") and not ml_cfg.get("enabled", False):
            ml_cfg = {**te_ml, **ml_cfg, "enabled": True}
        finetune = config.get("meta_finetune") or {}
        if finetune:
            ml_cfg["finetune"] = finetune
        return ml_cfg

    def attach_policy(self, policy: Any) -> None:
        self.meta.attach_policy(policy)

    def on_trade_closed(self, trajectory: list[dict], pnl_r: float) -> dict[str, Any]:
        is_win = pnl_r > 0
        self.trigger.add_result(is_win)
        regime = "unknown"
        if trajectory:
            regime = str(trajectory[-1].get("regime", self._last_regime))
        self.meta.add_trajectory(trajectory, regime=regime, pnl_r=pnl_r)
        result: dict[str, Any] = {"is_win": is_win, "pnl_r": pnl_r, "regime": regime}

        if self.trigger.should_adapt():
            logger.info(
                "适应触发：近%d笔胜率 %.1f%% < 阈值",
                len(self.trigger.recent_wins),
                self.trigger.winrate() * 100,
            )
            result["meta_update"] = self.meta.meta_update()
            result["adaptation_triggered"] = True
            result["finetune_queued"] = self.meta.note_adaptation_triggered()
        else:
            self.meta._adaptation_streak = max(0, self.meta._adaptation_streak - 1)
        return result

    def on_tick(self, regime: str | None = None) -> dict[str, Any] | None:
        if regime:
            self._last_regime = regime
        self._tick_steps += 1
        self.meta.record_step()
        if self.meta.should_run_scheduled_update():
            return self.meta.meta_update()
        return None

    def apply_action_bias(self, action: int, regime: str | None = None, n_actions: int = 6) -> int:
        """用元学习偏置微调离散动作（全局 + regime 条件）。"""
        reg = regime or self._last_regime
        bias = self.meta.action_bias(reg)
        if bias.size < n_actions or float(bias.max() - bias.min()) < 1e-4:
            return action
        adjusted = int(action)
        if action == 0 and bias[1] > bias[3] + 0.05:
            adjusted = 1
        elif action == 0 and bias[3] > bias[1] + 0.05:
            adjusted = 3
        elif action == 1 and bias[1] < bias[0] - 0.08:
            adjusted = 0
        elif action == 2 and bias[2] < bias[0] - 0.08:
            adjusted = 0
        return adjusted
