"""在线元学习（MAML 简化变体，CPU 友好）。"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    收集交易轨迹，周期性微调策略偏置或触发 PPO 继续训练。
    不阻塞主循环：meta_update 目标 < 0.5s。
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        state_dir: str | Path = "data/meta_learning",
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lr_meta = float(cfg.get("meta_learning_rate", 1e-4))
        self.alpha = float(cfg.get("inner_lr", 0.01))
        self.beta = float(cfg.get("reg_beta", 0.001))
        self.batch_size = int(cfg.get("meta_batch_size", 10))
        self.buffer_max = int(cfg.get("trajectory_buffer_size", 100))
        self.update_interval_steps = int(cfg.get("update_interval_steps", 50))
        self.max_param_delta_pct = float(cfg.get("max_param_delta_pct", 0.05))

        self.trajectory_buffer: deque[list[dict]] = deque(maxlen=self.buffer_max)
        self._step_counter = 0
        self._policy = None
        self._bias = np.zeros(6, dtype=np.float32)
        from zhulong.utils.paths import resolve_writable_data_path

        self.state_dir = (
            resolve_writable_data_path(state_dir)
            if not Path(state_dir).is_absolute()
            else Path(state_dir)
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def attach_policy(self, policy: Any) -> None:
        """可传入 stable_baselines3 PPO 或 RlAgent。"""
        self._policy = policy

    def add_trajectory(self, trajectory: list[dict]) -> None:
        if not self.enabled or not trajectory:
            return
        self.trajectory_buffer.append(list(trajectory))

    def record_step(self) -> None:
        self._step_counter += 1

    def should_run_scheduled_update(self) -> bool:
        return self.enabled and self._step_counter > 0 and self._step_counter % self.update_interval_steps == 0

    def action_bias(self) -> np.ndarray:
        return self._bias.copy()

    def meta_update(self, batch_size: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"skipped": True, "reason": "disabled"}
        bs = batch_size or self.batch_size
        if len(self.trajectory_buffer) < bs:
            return {"skipped": True, "reason": "insufficient_trajectories", "n": len(self.trajectory_buffer)}

        t0 = time.perf_counter()
        sampled = list(self.trajectory_buffer)[-bs:]
        rewards_by_action: dict[int, list[float]] = {i: [] for i in range(6)}
        for traj in sampled:
            for step in traj:
                a = int(step.get("action", 0))
                r = float(step.get("reward", 0.0))
                rewards_by_action.setdefault(a, []).append(r)

        mean_r = {a: (float(np.mean(rs)) if rs else 0.0) for a, rs in rewards_by_action.items()}
        old_bias = self._bias.copy()
        for a, mr in mean_r.items():
            if 0 <= a < len(self._bias):
                self._bias[a] += self.lr_meta * float(np.clip(mr, -1.0, 1.0))

        delta = float(np.linalg.norm(self._bias - old_bias))
        max_bias = float(np.max(np.abs(self._bias))) or 1e-9
        if delta / max_bias > self.max_param_delta_pct:
            self._bias = old_bias + (self._bias - old_bias) * (self.max_param_delta_pct * max_bias / delta)

        policy_updated = self._try_policy_touch(sampled)
        self._save_state()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = {
            "meta_loss": float(-sum(mean_r.values())),
            "mean_reward_by_action": mean_r,
            "bias_norm": float(np.linalg.norm(self._bias)),
            "policy_updated": policy_updated,
            "elapsed_ms": elapsed_ms,
            "trajectories": len(sampled),
        }
        logger.info("[MetaLearner] meta_update %.1fms bias_norm=%.4f", elapsed_ms, stats["bias_norm"])
        return stats

    def _try_policy_touch(self, trajectories: list[list[dict]]) -> bool:
        if self._policy is None:
            return False
        try:
            model = getattr(self._policy, "_model", None) or self._policy
            learn = getattr(model, "learn", None)
            if learn is None:
                return False
            learn(total_timesteps=min(64, len(trajectories) * 4), reset_num_timesteps=False)
            return True
        except Exception as ex:
            logger.debug("策略元触摸跳过: %s", ex)
            return False

    def _state_path(self) -> Path:
        return self.state_dir / "meta_state.json"

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.is_file():
            return
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            bias = blob.get("bias")
            if bias and len(bias) == 6:
                self._bias = np.asarray(bias, dtype=np.float32)
        except Exception as ex:
            logger.warning("元学习状态加载失败: %s", ex)

    def _save_state(self) -> None:
        try:
            self._state_path().write_text(
                json.dumps(
                    {
                        "bias": [float(x) for x in self._bias],
                        "step_counter": self._step_counter,
                        "buffer_len": len(self.trajectory_buffer),
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as ex:
            logger.warning("元学习状态保存失败: %s", ex)

    def export_trajectories_npz(self, path: str | Path) -> None:
        """供 weekly_finetune 使用。"""
        if not self.trajectory_buffer:
            return
        states, actions, rewards = [], [], []
        for traj in self.trajectory_buffer:
            for step in traj:
                if "state" in step:
                    states.append(step["state"])
                    actions.append(step.get("action", 0))
                    rewards.append(step.get("reward", 0.0))
        if not states:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
        )
