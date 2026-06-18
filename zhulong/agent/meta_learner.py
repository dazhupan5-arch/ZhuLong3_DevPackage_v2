"""在线元学习（MAML 简化变体 + Regime 条件偏置，CPU 友好）。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from zhulong.attribution.schema import normalize_regime

logger = logging.getLogger(__name__)

REGIMES = ("trend", "ranging", "volatile", "unknown")
N_ACTIONS = 6


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
        self.regime_lr = float(cfg.get("regime_learning_rate", self.lr_meta * 1.5))
        self.alpha = float(cfg.get("inner_lr", 0.01))
        self.beta = float(cfg.get("reg_beta", 0.001))
        self.batch_size = int(cfg.get("meta_batch_size", 10))
        self.buffer_max = int(cfg.get("trajectory_buffer_size", 100))
        self.update_interval_steps = int(cfg.get("update_interval_steps", 50))
        self.max_param_delta_pct = float(cfg.get("max_param_delta_pct", 0.05))
        self.regime_enabled = bool(cfg.get("regime_conditional", True))
        self.finetune_cfg = cfg.get("finetune") or {}

        self.trajectory_buffer: deque[dict[str, Any]] = deque(maxlen=self.buffer_max)
        self._step_counter = 0
        self._policy = None
        self._bias = np.zeros(N_ACTIONS, dtype=np.float32)
        self._regime_bias: dict[str, np.ndarray] = {r: np.zeros(N_ACTIONS, dtype=np.float32) for r in REGIMES}
        self._adaptation_streak = 0
        from zhulong.utils.paths import resolve_writable_data_path

        self.state_dir = (
            resolve_writable_data_path(state_dir)
            if not Path(state_dir).is_absolute()
            else Path(state_dir)
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def attach_policy(self, policy: Any) -> None:
        self._policy = policy

    def add_trajectory(
        self,
        trajectory: list[dict],
        *,
        regime: str = "unknown",
        pnl_r: float | None = None,
    ) -> None:
        if not self.enabled or not trajectory:
            return
        reward = float(pnl_r) if pnl_r is not None else 0.0
        steps = []
        for step in trajectory:
            s = dict(step)
            if reward != 0.0 and float(s.get("reward", 0.0)) == 0.0:
                s["reward"] = reward
            steps.append(s)
        self.trajectory_buffer.append(
            {"regime": normalize_regime(regime), "steps": steps, "pnl_r": reward}
        )

    def record_step(self) -> None:
        self._step_counter += 1

    def should_run_scheduled_update(self) -> bool:
        return self.enabled and self._step_counter > 0 and self._step_counter % self.update_interval_steps == 0

    def action_bias(self, regime: str | None = None) -> np.ndarray:
        out = self._bias.copy()
        if self.regime_enabled and regime:
            r = normalize_regime(regime)
            if r in self._regime_bias:
                out = out + self._regime_bias[r]
        return out

    def meta_update(self, batch_size: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"skipped": True, "reason": "disabled"}
        bs = batch_size or self.batch_size
        if len(self.trajectory_buffer) < bs:
            return {"skipped": True, "reason": "insufficient_trajectories", "n": len(self.trajectory_buffer)}

        t0 = time.perf_counter()
        sampled = list(self.trajectory_buffer)[-bs:]
        global_stats = self._update_bias(self._bias, self.lr_meta, sampled, regime_specific=False)
        regime_stats: dict[str, Any] = {}
        if self.regime_enabled:
            for reg in REGIMES:
                reg_trajs = [t for t in sampled if t.get("regime") == reg]
                if len(reg_trajs) >= max(2, bs // 4):
                    regime_stats[reg] = self._update_bias(
                        self._regime_bias[reg], self.regime_lr, reg_trajs, regime_specific=True
                    )

        policy_updated = self._try_policy_touch([t for x in sampled for t in x.get("steps", [])])
        self._save_state()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = {
            "meta_loss": global_stats.get("meta_loss", 0.0),
            "mean_reward_by_action": global_stats.get("mean_reward_by_action", {}),
            "regime_updates": regime_stats,
            "bias_norm": float(np.linalg.norm(self._bias)),
            "policy_updated": policy_updated,
            "elapsed_ms": elapsed_ms,
            "trajectories": len(sampled),
        }
        logger.info("[MetaLearner] meta_update %.1fms bias_norm=%.4f", elapsed_ms, stats["bias_norm"])
        return stats

    def _update_bias(
        self,
        bias: np.ndarray,
        lr: float,
        trajectories: list[dict],
        *,
        regime_specific: bool,
    ) -> dict[str, Any]:
        rewards_by_action: dict[int, list[float]] = {i: [] for i in range(N_ACTIONS)}
        for traj in trajectories:
            for step in traj.get("steps", []):
                if not isinstance(step, dict):
                    continue
                a = int(step.get("action", 0))
                r = float(step.get("reward", 0.0))
                rewards_by_action.setdefault(a, []).append(r)

        mean_r = {a: (float(np.mean(rs)) if rs else 0.0) for a, rs in rewards_by_action.items()}
        old = bias.copy()
        for a, mr in mean_r.items():
            if 0 <= a < len(bias):
                bias[a] += lr * float(np.clip(mr, -1.0, 1.0))

        delta = float(np.linalg.norm(bias - old))
        max_bias = float(np.max(np.abs(bias))) or 1e-9
        if delta / max_bias > self.max_param_delta_pct:
            bias[:] = old + (bias - old) * (self.max_param_delta_pct * max_bias / delta)

        return {
            "meta_loss": float(-sum(mean_r.values())),
            "mean_reward_by_action": mean_r,
            "regime_specific": regime_specific,
        }

    def note_adaptation_triggered(self) -> bool:
        self._adaptation_streak += 1
        need = int(self.finetune_cfg.get("consecutive_triggers", 3))
        if not self.finetune_cfg.get("enabled", False) or self._adaptation_streak < need:
            return False
        self._adaptation_streak = 0
        return self._enqueue_finetune()

    def _enqueue_finetune(self) -> bool:
        sym = str(self.finetune_cfg.get("symbol", "XAUUSD"))
        pending = self.state_dir / "finetune_pending.json"
        payload = {
            "symbol": sym,
            "requested_at": time.time(),
            "timesteps": int(self.finetune_cfg.get("timesteps", 5000)),
        }
        pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("[MetaLearner] finetune 已排队 symbol=%s", sym)
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "weekly_finetune.py"
        if self.finetune_cfg.get("auto_run", False) and script.is_file():
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        "--symbol",
                        sym,
                        "--timesteps",
                        str(payload["timesteps"]),
                    ],
                    cwd=str(script.parent.parent),
                )
                return True
            except Exception as ex:
                logger.warning("finetune 子进程启动失败: %s", ex)
        return True

    def _try_policy_touch(self, steps: list[dict]) -> bool:
        if self._policy is None or not steps:
            return False
        try:
            model = getattr(self._policy, "_model", None) or self._policy
            learn = getattr(model, "learn", None)
            if learn is None:
                return False
            learn(total_timesteps=min(64, len(steps) * 4), reset_num_timesteps=False)
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
            if bias and len(bias) == N_ACTIONS:
                self._bias = np.asarray(bias, dtype=np.float32)
            rb = blob.get("regime_bias") or {}
            for reg in REGIMES:
                if reg in rb and len(rb[reg]) == N_ACTIONS:
                    self._regime_bias[reg] = np.asarray(rb[reg], dtype=np.float32)
            self._adaptation_streak = int(blob.get("adaptation_streak", 0))
        except Exception as ex:
            logger.warning("元学习状态加载失败: %s", ex)

    def _save_state(self) -> None:
        try:
            self._state_path().write_text(
                json.dumps(
                    {
                        "bias": [float(x) for x in self._bias],
                        "regime_bias": {k: [float(x) for x in v] for k, v in self._regime_bias.items()},
                        "step_counter": self._step_counter,
                        "buffer_len": len(self.trajectory_buffer),
                        "adaptation_streak": self._adaptation_streak,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as ex:
            logger.warning("元学习状态保存失败: %s", ex)

    def export_trajectories_npz(self, path: str | Path) -> None:
        if not self.trajectory_buffer:
            return
        states, actions, rewards = [], [], []
        for traj in self.trajectory_buffer:
            for step in traj.get("steps", []):
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
