"""Gymnasium 强化学习交易环境。"""



from __future__ import annotations

import random
from typing import Any



import numpy as np

import pandas as pd



try:

    import gymnasium as gym

    from gymnasium import spaces

except ImportError:  # pragma: no cover

    gym = None  # type: ignore

    spaces = None  # type: ignore



from zhulong.agent.causal_inference import CounterfactualPredictor, CausalInference

from zhulong.agent.meta_learner import MetaLearner

from zhulong.agent.state_builder import (
    STATE_DIM,
    StateBuilder,
    gate_action_by_cognition,
    infer_regime_from_struct,
    primary_direction_from_probs,
)

from zhulong.agent.execution_composer import (
    limit_fill_on_bar,
    location_score,
    structure_entry_target,
)
from zhulong.agent.kn2_location_labels import LocationLabelConfig, compute_pos_in_range
from zhulong.agent.tick_brief import StructureSnapshot
from zhulong.agent.trader_memory import TraderMemory

from zhulong.strategies.indicators import atr_series





class TradingEnv(gym.Env if gym else object):  # type: ignore[misc]

    metadata = {"render_modes": []}



    def __init__(

        self,

        data: pd.DataFrame,

        struct_features: np.ndarray,

        knowledge_embeddings: np.ndarray,

        config: dict[str, Any] | None = None,

        symbol: str = "XAUUSD",

        scaler_path: str | None = None,

        exogenous_shocks: np.ndarray | None = None,

        meta_learner: MetaLearner | None = None,

        knowledge_probs: np.ndarray | None = None,

    ) -> None:

        if gym is None:

            raise ImportError("需要 gymnasium")

        super().__init__()

        self.config = config or {}

        self.symbol = symbol.upper()

        self.data = data.reset_index(drop=True)

        self.struct = np.asarray(struct_features, dtype=np.float32)

        self.emb = np.asarray(knowledge_embeddings, dtype=np.float32)

        n = min(len(self.data), len(self.struct), len(self.emb))

        self.data = self.data.iloc[:n].reset_index(drop=True)

        self.struct = self.struct[:n]

        self.emb = self.emb[:n]

        if knowledge_probs is not None:
            self.knowledge_probs = np.asarray(knowledge_probs, dtype=np.float32)[:n]
        else:
            self.knowledge_probs = None



        if exogenous_shocks is not None:

            self.exogenous = np.asarray(exogenous_shocks, dtype=np.float32)[:n]

        else:

            self.exogenous = np.zeros(n, dtype=np.float32)



        if "atr" not in self.data.columns:

            atr_df = atr_series(

                pd.DataFrame(

                    {

                        "high": self.data["high"],

                        "low": self.data["low"],

                        "close": self.data["close"],

                    }

                )

            )

            self.data = self.data.copy()

            self.data["atr"] = atr_df.bfill().fillna(self.data["close"] * 0.001)



        self.initial_balance = float(self.config.get("initial_balance", 10000))

        self.slippage = float(self.config.get("slippage", 0.1))

        self.sl_mult = float(self.config.get("stop_loss_atr_mult", 1.5))

        self.tp_mult = float(self.config.get("take_profit_atr_mult", 2.5))

        self.max_hold = int(self.config.get("max_hold_bars", 48))

        self.hold_penalty = float(self.config.get("hold_penalty", 0.0))

        self.open_reward_bonus = float(self.config.get("open_reward_bonus", 0.0))

        cog_cfg = self.config.get("cognition_align") or {}
        self.cognition_align_bonus = float(cog_cfg.get("same_direction_bonus", 0.02))
        self.cognition_align_penalty = float(cog_cfg.get("opposite_direction_penalty", 0.15))
        self.cognition_gate_actions = bool(cog_cfg.get("gate_actions", True))

        self.cost_scale = float(self.config.get("cost_scale", 1.0))

        self.max_trades_per_episode = int(self.config.get("max_trades_per_episode", 20))

        self.drawdown_penalty_coef = float(self.config.get("drawdown_penalty_coef", 0.1))
        self.drawdown_penalty_trigger = float(self.config.get("drawdown_penalty_trigger", 0.1))

        self.force_explore_steps = int(self.config.get("force_explore_steps", 0))

        self.unrealized_reward_scale = float(self.config.get("unrealized_reward_scale", 0.01))

        self.close_profit_bonus = float(self.config.get("close_profit_bonus", 0.1))

        self.simple3 = str(self.config.get("action_space", "")).lower() == "simple3"

        point_cost = (self.config.get("point_cost") or {}).get(self.symbol, 0.2)

        self.spread = float(point_cost)



        cf_cfg = self.config.get("counterfactual") or {}

        self.use_counterfactual_reward = bool(cf_cfg.get("enabled", False))

        causal_cfg = self.config.get("causal") or {}

        coef_path = causal_cfg.get("coef_path", "models/causal_coef.pkl")

        self._counterfactual = CounterfactualPredictor(

            CausalInference(coef_path, symbol=self.symbol) if self.use_counterfactual_reward else None

        )



        ml_cfg = self.config.get("meta_learning") or {}

        self.meta_learner = meta_learner

        if self.meta_learner is None and ml_cfg.get("enabled", False):

            self.meta_learner = MetaLearner(ml_cfg)



        self.state_builder = StateBuilder(scaler_path)

        self.memory = TraderMemory(int((self.config.get("trader_memory") or {}).get("max_len", 20)))



        n_actions = 3 if self.simple3 else 6

        self.action_space = spaces.Discrete(n_actions)

        self.observation_space = spaces.Box(

            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32

        )



        self.current_step = 0

        self.balance = self.initial_balance

        self.equity = self.initial_balance

        self.position = 0.0

        self.entry_price = 0.0

        self.entry_step = 0

        self.sl = 0.0

        self.tp = 0.0

        self.daily_pnl_r = 0.0

        self.peak_balance = self.initial_balance

        self.consecutive_losses = 0

        self.consecutive_wins = 0

        self.trades_this_episode = 0

        self._peak_equity = self.initial_balance

        self._peak_since_reward = self.initial_balance

        self._max_dd_since_reward = 0.0

        self.trades: list[dict] = []

        self.trajectory: list[dict] = []

        self._hold_exogenous_sum = 0.0

        self.peak_equity_this_trade = self.initial_balance

        self.max_drawdown_this_trade = 0.0

        self.total_env_steps = 0

        ep_cfg = self.config.get("execution_parity") or {}
        self.execution_parity = bool(ep_cfg.get("enabled", False))
        self.entry_quality_bonus = float(ep_cfg.get("entry_quality_bonus", 0.05))
        self.pending_expire_bars = int(ep_cfg.get("pending_expire_bars", 48))
        self.pending_expire_penalty = float(ep_cfg.get("pending_expire_penalty", 0.01))
        self._loc_cfg = LocationLabelConfig()
        self.pending_direction = 0
        self.pending_target = 0.0
        self.pending_size = 0.0
        self.pending_expire_step = 0
        self.signal_bar_close = 0.0



    def reset(self, *, seed: int | None = None, options: dict | None = None):

        super().reset(seed=seed)

        self.current_step = 0

        self.balance = self.initial_balance

        self.equity = self.initial_balance

        self.position = 0.0

        self.entry_price = 0.0

        self.entry_step = 0

        self.sl = 0.0

        self.tp = 0.0

        self.daily_pnl_r = 0.0

        self.peak_balance = self.initial_balance

        self.consecutive_losses = 0

        self.consecutive_wins = 0

        self.trades_this_episode = 0

        self._peak_equity = self.initial_balance

        self._peak_since_reward = self.initial_balance

        self._max_dd_since_reward = 0.0

        self.trades = []

        self.trajectory = []

        self._hold_exogenous_sum = 0.0

        self.peak_equity_this_trade = self.initial_balance

        self.max_drawdown_this_trade = 0.0

        self.total_env_steps = 0

        self.pending_direction = 0
        self.pending_target = 0.0
        self.pending_size = 0.0
        self.pending_expire_step = 0
        self.signal_bar_close = 0.0

        self.memory = TraderMemory(self.memory.max_len)

        return self._get_state(), {}



    def step(self, action: int):

        reward = 0.0

        done = False

        info: dict[str, Any] = {}

        self.total_env_steps += 1

        if self.force_explore_steps > 0 and self.total_env_steps <= self.force_explore_steps:

            action = int(self.action_space.sample())

        raw_action = int(action)
        if self.cognition_gate_actions:
            primary = self._primary_direction_at(self.current_step)
            action, _ = gate_action_by_cognition(raw_action, primary)



        if self.position != 0:

            self._record_trajectory_step(int(action), reward=0.0)

            self._hold_exogenous_sum += float(self.exogenous[min(self.current_step, len(self.exogenous) - 1)])



        close_reward = self._advance_pending_entry()

        reward += close_reward

        close_reward = self._execute_action(int(action))

        reward += close_reward



        self.current_step += 1

        if self.current_step >= len(self.data):

            done = True

            if self.position != 0:

                reward += self._close_position(float(self.data["close"].iloc[-1]), forced=True)

        else:

            sl_tp_reward = self._check_stop_loss_take_profit()

            reward += sl_tp_reward

            if self.position != 0 and self.current_step - self.entry_step >= self.max_hold:
                idx = min(self.current_step, len(self.data) - 1)
                mark = float(self.data["close"].iloc[idx])
                direction = 1 if self.position > 0 else -1
                unrealized = (mark - self.entry_price) * direction * abs(self.position)
                if unrealized <= 0:
                    reward += self._close_position(mark, forced=True)

            elif self.position != 0 and self.hold_penalty > 0:

                reward -= self.hold_penalty



        self._update_equity()

        if self.position != 0:

            idx = min(self.current_step, len(self.data) - 1)

            price = float(self.data["close"].iloc[idx])

            direction = 1 if self.position > 0 else -1

            unrealized = (price - self.entry_price) * direction * abs(self.position)

            reward += unrealized * self.unrealized_reward_scale

            current_equity = self.equity

            self.peak_equity_this_trade = max(self.peak_equity_this_trade, current_equity)

            if self.peak_equity_this_trade > 0:

                trade_dd = (self.peak_equity_this_trade - current_equity) / self.peak_equity_this_trade

                self.max_drawdown_this_trade = max(self.max_drawdown_this_trade, trade_dd)

        if self.drawdown_penalty_coef > 0:

            self._peak_equity = max(self._peak_equity, self.equity)

            if self._peak_equity > 0:

                drawdown = (self._peak_equity - self.equity) / self._peak_equity

                if drawdown > self.drawdown_penalty_trigger:

                    reward -= drawdown * self.drawdown_penalty_coef

        if self.max_trades_per_episode < 999 and self.trades_this_episode >= self.max_trades_per_episode:

            if self.position != 0:

                idx = min(self.current_step, len(self.data) - 1)

                reward += self._close_position(float(self.data["close"].iloc[idx]), forced=True)

            done = True

        state = self._get_state()

        return state, float(reward), done, False, info



    def _record_trajectory_step(self, action: int, reward: float) -> None:

        self.trajectory.append(

            {

                "step": self.current_step,

                "state": self._get_state().copy(),

                "action": action,

                "reward": reward,

            }

        )



    def _bar_time(self, idx: int):

        if "time" in self.data.columns:

            return self.data["time"].iloc[idx]

        return None



    def _get_state(self) -> np.ndarray:

        idx = min(max(self.current_step, 0), len(self.data) - 1)

        account = {

            "initial_balance": self.initial_balance,

            "balance": self.balance,

            "equity": self.equity,

            "position": self.position,

            "peak_balance": self.peak_balance,

            "daily_pnl_r": self.daily_pnl_r,

        }

        return self.state_builder.build(

            self.struct[idx],

            self.emb[idx],

            account,

            self.memory,

            self._bar_time(idx),

            cognition=self._cognition_ctx_at(idx),

        )



    def _knowledge_probs_at(self, idx: int) -> np.ndarray | None:
        if self.knowledge_probs is None:
            return None
        idx = min(max(idx, 0), len(self.knowledge_probs) - 1)
        row = np.asarray(self.knowledge_probs[idx], dtype=np.float32).reshape(-1)
        return row if row.size >= 3 else None

    def _primary_direction_at(self, idx: int) -> str:
        probs = self._knowledge_probs_at(idx)
        regime = infer_regime_from_struct(self.struct[idx])
        return primary_direction_from_probs(probs, regime)

    def _cognition_ctx_at(self, idx: int) -> dict[str, Any]:
        probs = self._knowledge_probs_at(idx)
        regime = infer_regime_from_struct(self.struct[idx])
        primary = primary_direction_from_probs(probs, regime)
        conf = 0.0
        if probs is not None and probs.size >= 3:
            conf = float(probs[2] if primary == "long" else probs[0] if primary == "short" else probs[1])
        return {
            "calibrated_probs": probs,
            "regime": regime,
            "confidence": conf,
            "should_trade": primary in ("long", "short") and conf >= 0.42,
        }



    def _cost(self, price: float) -> float:

        return (self.spread + self.slippage) * self.cost_scale



    def _position_size(self, action: int) -> float:

        if self.simple3:

            if action == 1:

                return 1.0

            if action == 2:

                return -1.0

            return 0.0

        if action in (1, 2):

            return 1.0 if action == 1 else -1.0

        if action in (3, 4):

            frac = 0.5 if action == 3 else 1.0

            return -frac

        return 0.0



    def _struct_snapshot_at(self, idx: int) -> StructureSnapshot:
        row = np.asarray(self.struct[idx], dtype=np.float32).reshape(-1)
        return StructureSnapshot(
            vector=row.tolist(),
            m5_trend=float(row[0]) if row.size > 0 else 0.0,
            support_dist_atr=float(row[3]) if row.size > 3 else 0.0,
            resistance_dist_atr=float(row[4]) if row.size > 4 else 0.0,
        )

    def _pos_in_range_at(self, idx: int) -> float:
        start = max(0, idx - 11)
        closes = self.data["close"].iloc[start : idx + 1].values.astype(np.float64)
        if len(closes) < 2:
            return 0.5
        pos_arr = compute_pos_in_range(closes.astype(np.float32))
        return float(pos_arr[-1])

    def _entry_quality_reward(self, direction: str, pos: float) -> float:
        loc = location_score(direction, pos, self._loc_cfg)
        return self.entry_quality_bonus * loc

    def _clear_pending(self) -> None:
        self.pending_direction = 0
        self.pending_target = 0.0
        self.pending_size = 0.0
        self.pending_expire_step = 0
        self.signal_bar_close = 0.0

    def _advance_pending_entry(self) -> float:
        if not self.execution_parity or self.position != 0 or self.pending_direction == 0:
            return 0.0
        idx = self.current_step
        if idx >= len(self.data):
            return 0.0
        reward = 0.0
        direction = "long" if self.pending_direction > 0 else "short"
        high = float(self.data["high"].iloc[idx])
        low = float(self.data["low"].iloc[idx])
        close = float(self.data["close"].iloc[idx])
        fill = limit_fill_on_bar(direction, self.pending_target, high, low, close)
        if fill is not None:
            reward += self._fill_pending_at(fill, idx)
        elif idx >= self.pending_expire_step:
            reward -= self.pending_expire_penalty
            self._clear_pending()
        return reward

    def _fill_pending_at(self, fill: float, idx: int) -> float:
        target_size = float(self.pending_size or (1.0 if self.pending_direction > 0 else -1.0))
        direction = "long" if target_size > 0 else "short"
        pos = self._pos_in_range_at(idx)
        reward = self._open_position_at(fill, target_size, idx)
        reward += self._entry_quality_reward(direction, pos)
        self._clear_pending()
        return reward

    def _open_position_at(self, fill: float, target: float, idx: int) -> float:
        atr = float(self.data["atr"].iloc[idx])
        cost = self._cost(fill)
        fill = fill + cost * (1 if target > 0 else -1)
        was_flat = self.position == 0
        self.position = target
        self.entry_price = fill
        self.entry_step = idx
        self.trajectory = []
        self._hold_exogenous_sum = float(self.exogenous[min(idx, len(self.exogenous) - 1)])
        if target > 0:
            self.sl = fill - self.sl_mult * atr
            self.tp = fill + self.tp_mult * atr
        else:
            self.sl = fill + self.sl_mult * atr
            self.tp = fill - self.tp_mult * atr
        self.peak_equity_this_trade = self.equity
        self.max_drawdown_this_trade = 0.0
        reward = 0.0
        if was_flat and target != 0:
            reward += self.open_reward_bonus
            primary = self._primary_direction_at(idx)
            if primary == "long" and target > 0:
                reward += self.cognition_align_bonus
            elif primary == "short" and target < 0:
                reward += self.cognition_align_bonus
            elif primary in ("long", "short"):
                reward -= self.cognition_align_penalty
        return reward

    def _execute_action(self, action: int) -> float:

        if action == 0:

            return 0.0

        if not self.simple3 and action == 5:

            if self.position == 0:

                return 0.0

            price = float(self.data["close"].iloc[self.current_step])

            return self._close_position(price)



        target = self._position_size(action)

        price = float(self.data["close"].iloc[self.current_step])

        reward = 0.0



        if self.position != 0 and np.sign(self.position) != np.sign(target):

            reward += self._close_position(price)



        if target == 0:

            return reward



        if self.position != 0 and np.sign(self.position) == np.sign(target):

            return reward



        was_flat = self.position == 0

        if was_flat and self.trades_this_episode >= self.max_trades_per_episode:

            return reward - 0.01

        atr = float(self.data["atr"].iloc[self.current_step])

        if self.execution_parity and was_flat:
            direction = "long" if target > 0 else "short"
            snap = self._struct_snapshot_at(self.current_step)
            pos = self._pos_in_range_at(self.current_step)
            loc = location_score(direction, pos, self._loc_cfg)
            entry_target = structure_entry_target(
                direction, snap, price, atr, loc_score=loc
            )
            high = float(self.data["high"].iloc[self.current_step])
            low = float(self.data["low"].iloc[self.current_step])
            fill_px = limit_fill_on_bar(direction, entry_target, high, low, price)
            if fill_px is not None:
                reward += self._open_position_at(fill_px, target, self.current_step)
                reward += self._entry_quality_reward(direction, pos)
                return reward
            self.pending_direction = 1 if target > 0 else -1
            self.pending_target = entry_target
            self.pending_size = target
            self.pending_expire_step = self.current_step + self.pending_expire_bars
            self.signal_bar_close = price
            return reward - 0.002

        fill = price + self._cost(price) * (1 if target > 0 else -1)

        self.position = target

        self.entry_price = fill

        self.entry_step = self.current_step

        self.trajectory = []

        self._hold_exogenous_sum = float(self.exogenous[min(self.current_step, len(self.exogenous) - 1)])

        if target > 0:

            self.sl = fill - self.sl_mult * atr

            self.tp = fill + self.tp_mult * atr

        else:

            self.sl = fill + self.sl_mult * atr

            self.tp = fill - self.tp_mult * atr

        self.peak_equity_this_trade = self.equity

        self.max_drawdown_this_trade = 0.0

        if was_flat and target != 0:

            reward += self.open_reward_bonus
            primary = self._primary_direction_at(self.current_step)
            opened_long = target > 0
            if primary == "long" and opened_long:
                reward += self.cognition_align_bonus
            elif primary == "short" and target < 0:
                reward += self.cognition_align_bonus
            elif primary in ("long", "short"):
                reward -= self.cognition_align_penalty

        return reward



    def _close_position(self, price: float, forced: bool = False) -> float:

        if self.position == 0:

            return 0.0

        direction = 1 if self.position > 0 else -1

        fill = price - self._cost(price) * direction

        pnl = (fill - self.entry_price) * direction * abs(self.position)

        pnl_r = pnl / max(self.initial_balance, 1e-9)

        hold_bars = max(self.current_step - self.entry_step, 1)

        if pnl_r < 0:

            self.consecutive_losses = min(self.consecutive_losses + 1, 10)

            self.consecutive_wins = 0

        else:

            self.consecutive_wins = min(self.consecutive_wins + 1, 10)

            self.consecutive_losses = 0

        reward = pnl_r

        if pnl_r > 0:

            reward += self.close_profit_bonus

        luck_r = 0.0

        if self.use_counterfactual_reward:

            luck_r = self._counterfactual.luck_pnl_r(

                direction=float(direction),

                entry_price=self.entry_price,

                position_frac=abs(self.position),

                initial_balance=self.initial_balance,

                exogenous_sum=self._hold_exogenous_sum,

                hold_bars=hold_bars,

            )

            reward = self._counterfactual.causal_reward(pnl_r, luck_r) + (reward - pnl_r)

        self.balance += pnl

        self.daily_pnl_r += pnl_r

        self.equity = self.balance

        self.trades_this_episode += 1

        self.memory.add_trade(pnl_r, int(self.current_step))

        self.trades.append(

            {

                "pnl_r": pnl_r,

                "causal_reward": float(reward),

                "luck_r": luck_r,

                "entry_step": self.entry_step,

                "exit_step": self.current_step,

                "entry_price": self.entry_price,

                "exit_price": price,

                "pnl": pnl,

                "direction": direction,

                "step": self.current_step,

                "forced": forced,

            }

        )



        if self.trajectory:

            self.trajectory[-1]["reward"] = float(reward)

            if self.meta_learner is not None:
                regime = "unknown"
                if self.trajectory and "regime" in self.trajectory[-1]:
                    regime = str(self.trajectory[-1].get("regime", "unknown"))
                self.meta_learner.add_trajectory(list(self.trajectory), regime=regime, pnl_r=float(reward))



        self.position = 0.0

        self.entry_price = 0.0

        self.sl = 0.0

        self.tp = 0.0

        self.trajectory = []

        self._hold_exogenous_sum = 0.0

        self.peak_equity_this_trade = self.equity

        self.max_drawdown_this_trade = 0.0

        self.peak_balance = max(self.peak_balance, self.equity)

        self._peak_since_reward = self.equity

        self._max_dd_since_reward = 0.0

        return reward



    def _check_stop_loss_take_profit(self) -> float:

        if self.position == 0 or self.current_step <= 0:

            return 0.0

        row = self.data.iloc[self.current_step]

        high = float(row["high"])

        low = float(row["low"])

        if self.position > 0:

            if low <= self.sl:

                return self._close_position(self.sl)

            if high >= self.tp:

                return self._close_position(self.tp)

        else:

            if high >= self.sl:

                return self._close_position(self.sl)

            if low <= self.tp:

                return self._close_position(self.tp)

        return 0.0



    def _update_equity(self) -> None:

        if self.position == 0:

            self.equity = self.balance

        else:

            price = float(self.data["close"].iloc[min(self.current_step, len(self.data) - 1)])

            direction = 1 if self.position > 0 else -1

            unreal = (price - self.entry_price) * direction * abs(self.position)

            self.equity = self.balance + unreal

        self.peak_balance = max(self.peak_balance, self.equity)

        peak = max(self._peak_since_reward, self.equity)

        self._peak_since_reward = peak

        dd = (peak - self.equity) / max(peak, 1e-9)

        self._max_dd_since_reward = max(self._max_dd_since_reward, dd)


class ForcedOpenExplorationWrapper(gym.Wrapper if gym else object):  # type: ignore[misc]
    """训练早期以一定概率强制开仓，缓解策略退化为永远 hold。"""

    def __init__(
        self,
        env,
        *,
        explore_steps: int = 20000,
        open_prob: float = 0.3,
    ) -> None:
        if gym is None:
            raise ImportError("需要 gymnasium")
        super().__init__(env)
        self.explore_steps = int(explore_steps)
        self.open_prob = float(open_prob)
        self._total_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._total_steps < self.explore_steps and random.random() < self.open_prob:
            inner = self.env
            while hasattr(inner, "env"):
                inner = inner.env
            primary = "flat"
            if hasattr(inner, "_primary_direction_at"):
                primary = inner._primary_direction_at(getattr(inner, "current_step", 0))
            if primary == "long":
                action = 1
            elif primary == "short":
                action = random.choice([2, 3, 4]) if not getattr(inner, "simple3", False) else 2
            elif getattr(inner, "simple3", False):
                action = random.choice([1, 2])
            else:
                action = random.choice([1, 2, 3, 4])
        self._total_steps += 1
        return self.env.step(action)


