"""KN 2.0 端到端训练环境。

KN 作为全权决策者，在历史数据上做完整的交易生命周期：
  无仓 → 开仓 → 持仓管理 → 出局 → 计算 ROI

训练目标：最大化 ROI，最小化最大回撤。

和 PPO TradingEnv 的区别：
  - KN 替代 PPO 做决策（action + size + SL/TP 都由 KN 输出）
  - GRU 隐藏状态跨 bar 延续
  - Triple Barrier 作为事后验证标签
  - 完整持仓管理：减仓、移动止损、浮盈回撤保护
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from zhulong.agent.state_builder import STATE_DIM, StateBuilder, infer_regime_from_struct
from zhulong.agent.trader_memory import TraderMemory
from zhulong.strategies.indicators import atr_series

ACTION_NAMES = ["hold", "long", "short", "short_50", "short_100", "close"]


class TradingEnvKN2(gym.Env if gym else object):
    """
    KN 2.0 训练环境 —— GRU 作为策略网络，端到端优化 ROI。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        market_features: np.ndarray,     # (n_bars, 98) V14+struct
        position_states: np.ndarray,     # (n_bars, 6) or zeros
        config: dict[str, Any] | None = None,
        symbol: str = "XAUUSD",
        scaler_path: str | None = None,
    ) -> None:
        if gym is None:
            raise ImportError("需要 gymnasium")

        super().__init__()
        self.config = config or {}
        self.symbol = symbol.upper()
        self.data = data.reset_index(drop=True)

        n = min(len(self.data), len(market_features), len(position_states))
        self.data = self.data.iloc[:n].reset_index(drop=True)
        self.market_features = np.asarray(market_features[:n], dtype=np.float32)
        self.position_states = np.asarray(position_states[:n], dtype=np.float32)

        # ATR
        if "atr" not in self.data.columns:
            atr_df = atr_series(pd.DataFrame({
                "high": self.data["high"],
                "low": self.data["low"],
                "close": self.data["close"],
            }))
            self.data = self.data.copy()
            self.data["atr"] = atr_df.bfill().fillna(self.data["close"] * 0.001)

        # 配置
        self.initial_balance = float(self.config.get("initial_balance", 10000))
        self.slippage = float(self.config.get("slippage", 0.1))
        self.max_hold = int(self.config.get("max_hold_bars", 48))
        self.hold_penalty = float(self.config.get("hold_penalty", 0.0001))
        self.max_trades_per_episode = int(self.config.get("max_trades_per_episode", 20))

        point_cost = (self.config.get("point_cost") or {}).get(self.symbol, 0.2)
        self.spread = float(point_cost)

        # 风险参数
        self.drawdown_penalty_coef = float(self.config.get("drawdown_penalty_coef", 0.5))
        self.drawdown_penalty_trigger = float(self.config.get("drawdown_penalty_trigger", 0.05))
        self.unrealized_reward_scale = float(self.config.get("unrealized_reward_scale", 0.01))
        self.close_profit_bonus = float(self.config.get("close_profit_bonus", 0.1))

        # 完整持仓管理
        self.profit_protect_pct = float(self.config.get("profit_protect_pct", 0.5))
        self.trail_stop_atr = float(self.config.get("trail_stop_atr", 1.0))

        # 状态构建器
        self.state_builder = StateBuilder(scaler_path)
        # TraderMemory for the environment's internal tracking
        self.memory = TraderMemory(int((self.config.get("trader_memory") or {}).get("max_len", 20)))

        # 动作空间: hold/long/short/short_50/short_100/close + 减仓 + 移动止损 + trail
        # 简化：使用和 PPO 相同的 6 动作
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance
        self.peak_equity = self.initial_balance

        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.sl = 0.0
        self.tp = 0.0
        self.kn_sl_mult = 1.5
        self.kn_tp_mult = 2.0

        # 持仓管理状态
        self.trade_direction = 0
        self.bars_held = 0
        self.peak_float_pnl_pct = 0.0
        self.max_adverse_pct = 0.0

        self.trades_this_episode = 0
        self.trades: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []

        self.memory = TraderMemory(self.memory.max_len)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_state(), {}

    def step(self, kn_decision: dict[str, Any]):
        """
        KN 决策驱动一步。

        Args:
            kn_decision: dict from KN2Inference.predict() with:
                action, position_size, sl_atr_mult, tp_atr_mult,
                confidence, should_trade

        Returns:
            (state, reward, done, truncated, info)
        """
        reward = 0.0
        done = False
        info: dict[str, Any] = {}

        action = int(kn_decision.get("action", 0))
        position_size = float(kn_decision.get("position_size", 1.0))
        sl_mult = float(kn_decision.get("sl_atr_mult", 1.5))
        tp_mult = float(kn_decision.get("tp_atr_mult", 2.0))
        confidence = float(kn_decision.get("confidence", 0.5))
        should_trade = bool(kn_decision.get("should_trade", False))

        idx = self.current_step
        price = float(self.data["close"].iloc[idx])
        atr = float(self.data["atr"].iloc[idx])

        # ---- 持仓中：更新持仓状态 ----
        if self.position != 0:
            self.bars_held += 1

            # 计算浮盈
            if self.trade_direction > 0:  # 多头
                float_pnl = (price - self.entry_price) * abs(self.position)
            else:  # 空头
                float_pnl = (self.entry_price - price) * abs(self.position)

            float_pnl_pct = float_pnl / max(self.equity, 1e-9)
            self.peak_float_pnl_pct = max(self.peak_float_pnl_pct, float_pnl_pct)

            # 浮盈回撤惩罚
            if self.peak_float_pnl_pct > 0.01 and float_pnl_pct < self.peak_float_pnl_pct * self.profit_protect_pct:
                reward -= (self.peak_float_pnl_pct - float_pnl_pct) * 5.0

            # 持仓过久惩罚
            if self.bars_held > self.max_hold:
                reward -= 0.01 * (self.bars_held - self.max_hold)

            # 检查 SL/TP（使用 KN 指定的或默认的）
            effective_sl_mult = max(sl_mult, self.kn_sl_mult)
            effective_tp_mult = max(tp_mult, self.kn_tp_mult)

            if self.trade_direction > 0:
                sl_price = self.entry_price - effective_sl_mult * atr
                tp_price = self.entry_price + effective_tp_mult * atr
                if price <= sl_price:
                    reward += self._close_position(price, forced=True, reason="sl")
                elif price >= tp_price:
                    reward += self._close_position(price, forced=True, reason="tp")
            else:
                sl_price = self.entry_price + effective_sl_mult * atr
                tp_price = self.entry_price - effective_tp_mult * atr
                if price >= sl_price:
                    reward += self._close_position(price, forced=True, reason="sl")
                elif price <= tp_price:
                    reward += self._close_position(price, forced=True, reason="tp")

            # 超时强平
            if self.position != 0 and self.bars_held >= self.max_hold:
                reward += self._close_position(price, forced=True, reason="timeout")

            # 对冲动作：KN 判断该反向或平仓
            if self.position != 0:
                if action == 5:  # close
                    reward += self._close_position(price, forced=True, reason="kn_close")
                elif (self.trade_direction > 0 and action in (2, 3, 4)) or \
                     (self.trade_direction < 0 and action == 1):
                    # 反向开仓：先平
                    reward += self._close_position(price, forced=True, reason="kn_reverse")
                    # 然后开新仓（继续执行下面的开仓逻辑）
                elif action in (3, 4) and self.trade_direction < 0:
                    # 加仓/调整（short_50, short_100）
                    pass  # 当前简化处理，不加仓

        # ---- 开仓逻辑 ----
        if self.position == 0 and should_trade and action in (1, 2):
            if self.trades_this_episode >= self.max_trades_per_episode:
                reward -= 0.01
            else:
                direction = 1 if action == 1 else -1
                fill = price + self._cost(price) * (1 if direction > 0 else -1)

                self.position = direction * position_size
                self.entry_price = fill
                self.entry_step = self.current_step
                self.kn_sl_mult = sl_mult
                self.kn_tp_mult = tp_mult
                self.trade_direction = direction
                self.bars_held = 0
                self.peak_float_pnl_pct = 0.0
                self.max_adverse_pct = 0.0

                if direction > 0:
                    self.sl = fill - sl_mult * atr
                    self.tp = fill + tp_mult * atr
                else:
                    self.sl = fill + sl_mult * atr
                    self.tp = fill - tp_mult * atr

                self.trades_this_episode += 1
                # 开仓小额正向奖励
                reward += 0.001 * confidence

        # ---- 更新权益 ----
        if self.position != 0:
            if self.trade_direction > 0:
                unrealized = (price - self.entry_price) * abs(self.position)
            else:
                unrealized = (self.entry_price - price) * abs(self.position)
            self.equity = self.balance + unrealized
        else:
            self.equity = self.balance

        self.peak_equity = max(self.peak_equity, self.equity)

        # 回撤惩罚
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > self.drawdown_penalty_trigger:
                reward -= drawdown * self.drawdown_penalty_coef

        # 持仓期间的未实现收益计入奖励
        if self.position != 0:
            if self.trade_direction > 0:
                unrealized_pnl = (price - self.entry_price) * abs(self.position)
            else:
                unrealized_pnl = (self.entry_price - price) * abs(self.position)
            reward += (unrealized_pnl / max(self.initial_balance, 1e-9)) * self.unrealized_reward_scale

        # ---- 下一步 ----
        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
            if self.position != 0:
                reward += self._close_position(
                    float(self.data["close"].iloc[-1]), forced=True, reason="episode_end"
                )

        if self.balance <= self.initial_balance * 0.5:
            done = True

        state = self._get_state()
        return state, float(reward), done, False, info

    def _close_position(
        self, price: float, forced: bool = False, reason: str = ""
    ) -> float:
        if self.position == 0:
            return 0.0

        direction = 1 if self.position > 0 else -1
        fill = price - self._cost(price) * direction
        pnl = (fill - self.entry_price) * direction * abs(self.position)
        pnl_r = pnl / max(self.initial_balance, 1e-9)

        self.balance += pnl
        self.equity = self.balance
        self.peak_balance = max(self.peak_balance, self.balance)

        reward = pnl_r

        if pnl_r > 0:
            reward += self.close_profit_bonus
            self.memory.add_trade(pnl_r, "")
        else:
            self.memory.add_trade(pnl_r, "")
            # 亏损交易惩罚
            reward -= min(abs(pnl_r), 0.05)

        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": fill,
            "pnl": pnl,
            "pnl_r": pnl_r,
            "direction": self.trade_direction,
            "bars_held": self.bars_held,
            "reason": reason,
            "kn_sl_mult": self.kn_sl_mult,
            "kn_tp_mult": self.kn_tp_mult,
        })

        # 重置持仓状态
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.sl = 0.0
        self.tp = 0.0
        self.trade_direction = 0
        self.bars_held = 0
        self.peak_float_pnl_pct = 0.0
        self.max_adverse_pct = 0.0
        self.kn_sl_mult = 1.5
        self.kn_tp_mult = 2.0

        return reward

    def _cost(self, price: float) -> float:
        return self.spread + self.slippage

    def _get_position_state(self) -> np.ndarray:
        """获取当前持仓状态编码。"""
        from zhulong.agent.knowledge_net_kn2 import encode_position_state

        if self.position == 0:
            return encode_position_state()

        if self.trade_direction > 0:
            float_pnl = (float(self.data["close"].iloc[self.current_step]) - self.entry_price)
        else:
            float_pnl = (self.entry_price - float(self.data["close"].iloc[self.current_step]))

        float_pnl_pct = float(float_pnl / max(self.equity, 1e-9))
        peak_pct = max(self.peak_float_pnl_pct, float_pnl_pct)

        return encode_position_state(
            direction=float(self.trade_direction),
            hold_bars=self.bars_held,
            float_pnl_pct=float_pnl_pct,
            max_favorable_pct=peak_pct,
            max_adverse_pct=self.max_adverse_pct,
            max_hold_bars=self.max_hold,
        )

    def _get_state(self) -> np.ndarray:
        idx = min(max(self.current_step, 0), len(self.data) - 1)
        account = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "peak_balance": self.peak_balance,
            "daily_pnl_r": 0.0,
        }

        # 使用 struct（从 market_features 中提取结构部分，简化）
        struct = self.market_features[idx, 68:98] if self.market_features.shape[1] >= 98 else np.zeros(30, dtype=np.float32)
        if struct.shape[0] < 30:
            struct = np.pad(struct, (0, 30 - struct.shape[0]))

        # 使用零嵌入（KN 2.0 不需要预计算的嵌入）
        emb = np.zeros(32, dtype=np.float32)

        regime = infer_regime_from_struct(struct)

        return self.state_builder.build(
            struct,
            emb,
            account,
            self.memory,
            self._bar_time(idx),
            cognition={
                "calibrated_probs": np.array([0.33, 0.34, 0.33], dtype=np.float32),
                "regime": regime,
                "confidence": 0.5,
                "should_trade": False,
            },
        )

    def _bar_time(self, idx: int):
        if "time" in self.data.columns:
            return self.data["time"].iloc[idx]
        return None

    # ===== PPO 兼容接口（用于 KN 替代 PPO 训练） =====

    def kn_step(self, kn_model, hidden_state):
        """
        一步环境交互：KN 模型决策 → 环境执行 → 返回 (state, reward, done, hidden, info)。

        Args:
            kn_model: TraderKnowledgeGRU 实例
            hidden_state: 上一步 GRU 隐藏状态

        Returns:
            (state, reward, done, next_hidden, info)
        """
        torch, _ = self._get_torch()
        idx = self.current_step

        mf = torch.tensor(self.market_features[idx:idx+1])
        ps = torch.tensor(self.get_position_state_np().reshape(1, -1))

        with torch.no_grad():
            outputs = kn_model(mf, hidden_state, ps)

        decision = {
            "action": int(outputs["action_logits"].argmax(dim=-1)[0]),
            "position_size": float(outputs["position_size"][0, 0]),
            "sl_atr_mult": float(outputs["sl_atr_mult"][0, 0]),
            "tp_atr_mult": float(outputs["tp_atr_mult"][0, 0]),
            "confidence": float(outputs["confidence"][0, 0]),
            "should_trade": bool(outputs["should_trade_prob"][0, 0] > 0.5),
        }

        state, reward, done, _, info = self.step(decision)
        return state, reward, done, outputs["hidden"], info

    def get_position_state_np(self) -> np.ndarray:
        return self._get_position_state()

    @staticmethod
    def _get_torch():
        from zhulong.agent.knowledge_net_kn2 import _ensure_torch
        torch, _ = _ensure_torch()
        return torch, None


# ==============================================================================
# Triple Barrier 事后验证标签
# ==============================================================================


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _kn2_labels_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        tp_atr_mult: float,
        sl_atr_mult: float,
        max_hold_bars: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = close.shape[0]
        actions = np.zeros(n, dtype=np.int32)
        sizes = np.zeros(n, dtype=np.float32)
        should_trade = np.zeros(n, dtype=np.float32)
        limit = n - max_hold_bars
        for t in prange(limit):
            entry_price = close[t]
            atr_t = atr[t]
            upper = entry_price + tp_atr_mult * atr_t
            lower = entry_price - sl_atr_mult * atr_t
            short_tp = entry_price - tp_atr_mult * atr_t
            short_sl = entry_price + sl_atr_mult * atr_t
            horizon = t + max_hold_bars + 1
            long_hit_tp = False
            long_hit_sl = False
            short_hit_tp = False
            short_hit_sl = False
            for fwd in range(t + 1, horizon):
                h = high[fwd]
                l = low[fwd]
                if not long_hit_tp and h >= upper:
                    long_hit_tp = True
                if not long_hit_sl and l <= lower:
                    long_hit_sl = True
                if not short_hit_tp and l <= short_tp:
                    short_hit_tp = True
                if not short_hit_sl and h >= short_sl:
                    short_hit_sl = True
                if (long_hit_tp and long_hit_sl) or (short_hit_tp and short_hit_sl):
                    break
            if long_hit_tp and not long_hit_sl:
                actions[t] = 1
                sizes[t] = 1.0
                should_trade[t] = 1.0
            if short_hit_tp and not short_hit_sl:
                if actions[t] == 1:
                    if tp_atr_mult / sl_atr_mult > tp_atr_mult / sl_atr_mult:
                        actions[t] = 2
                else:
                    actions[t] = 2
                sizes[t] = 1.0
                should_trade[t] = 1.0
        return actions, sizes, should_trade

    _KN2_LABELS_FAST = True
except ImportError:
    _KN2_LABELS_FAST = False


def generate_kn2_training_labels(
    data: pd.DataFrame,
    market_features: np.ndarray,
    *,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    max_hold_bars: int = 48,
    min_rr_ratio: float = 1.5,
    progress_every: int = 0,
) -> dict[str, np.ndarray]:
    """
    从历史数据生成 KN 2.0 训练标签。

    对每根 K 线：
      1. 模拟在此处开仓（多和空两个方向）
      2. 用 Triple Barrier 判断哪个屏障先触及
      3. 记录最优 action + SL/TP + position_size

    Returns:
        dict with keys: action, position_size, sl_atr_mult, tp_atr_mult, should_trade
    """
    n = len(data)
    close = data["close"].values
    high = data["high"].values if "high" in data.columns else close
    low = data["low"].values if "low" in data.columns else close

    atr = data["atr"].values if "atr" in data.columns else np.full(n, close[0] * 0.001)

    sl_mults = np.full(n, sl_atr_mult, dtype=np.float32)
    tp_mults = np.full(n, tp_atr_mult, dtype=np.float32)

    if _KN2_LABELS_FAST:
        actions, sizes, should_trade = _kn2_labels_numba(
            close.astype(np.float64),
            high.astype(np.float64),
            low.astype(np.float64),
            atr.astype(np.float64),
            float(tp_atr_mult),
            float(sl_atr_mult),
            int(max_hold_bars),
        )
        return {
            "action": actions,
            "position_size": sizes,
            "sl_atr_mult": sl_mults,
            "tp_atr_mult": tp_mults,
            "should_trade": should_trade,
        }

    actions = np.zeros(n, dtype=np.int32)  # 默认 hold
    sizes = np.zeros(n, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)

    for t in range(n - max_hold_bars):
        entry_price = close[t]
        atr_t = atr[t]

        # 上轨止盈 / 下轨止损
        upper = entry_price + tp_atr_mult * atr_t
        lower = entry_price - sl_atr_mult * atr_t

        horizon = min(t + max_hold_bars + 1, n)
        long_hit_tp = False
        long_hit_sl = False
        short_hit_tp = False
        short_hit_sl = False

        if progress_every and t > 0 and t % progress_every == 0:
            print(f"  kn2 labels {t:,}/{n - max_hold_bars:,} ({100.0 * t / max(n - max_hold_bars, 1):.1f}%)")

        for fwd in range(t + 1, horizon):
            h, l = high[fwd], low[fwd]

            if not long_hit_tp and h >= upper:
                long_hit_tp = True
            if not long_hit_sl and l <= lower:
                long_hit_sl = True
            if not short_hit_tp and l <= entry_price - tp_atr_mult * atr_t:
                short_hit_tp = True
            if not short_hit_sl and h >= entry_price + sl_atr_mult * atr_t:
                short_hit_sl = True

            if long_hit_tp and long_hit_sl:
                break
            if short_hit_tp and short_hit_sl:
                break

        # 判断做多
        if long_hit_tp and not long_hit_sl:
            actions[t] = 1  # long
            sizes[t] = 1.0
            should_trade[t] = 1.0
        elif long_hit_sl and not long_hit_tp:
            pass  # 不做多
        # 否则中立

        # 判断做空
        if short_hit_tp and not short_hit_sl:
            if actions[t] == 1:
                # 多空都成立，选 RR 更高的
                long_rr = tp_atr_mult / sl_atr_mult
                short_rr = tp_atr_mult / sl_atr_mult
                if short_rr > long_rr:
                    actions[t] = 2  # short
            else:
                actions[t] = 2  # short
            sizes[t] = 1.0
            should_trade[t] = 1.0

    return {
        "action": actions,
        "position_size": sizes,
        "sl_atr_mult": sl_mults,
        "tp_atr_mult": tp_mults,
        "should_trade": should_trade,
    }


# ==============================================================================
# PPO + KN 2.0 组合训练
# ==============================================================================


def train_kn2_with_ppo(
    env: TradingEnvKN2,
    kn_model,
    ppo_model=None,
    *,
    total_timesteps: int = 100_000,
    sequence_length: int = 64,
    log_interval: int = 1000,
) -> dict[str, Any]:
    """
    用 PPO 训练 KN 2.0 作为策略网络。

    KN 2.0 替代 PPO 的 actor，直接输出 action 分布。
    PPO 的 value 网络学习评估状态价值。
    训练目标：最大化 episode 收益。
    """
    torch, nn = _get_torch_local()

    optimizer = torch.optim.AdamW(kn_model.parameters(), lr=3e-4)
    total_rewards: list[float] = []
    episode_rewards: list[float] = []

    obs, _ = env.reset()
    hidden = None
    episode_reward = 0.0
    episode_steps = 0

    log_buffer: list[dict] = []

    for step in range(total_timesteps):
        idx = env.current_step
        mf = torch.tensor(env.market_features[idx:idx+1])
        ps = torch.tensor(env.get_position_state_np().reshape(1, -1))

        # KN 前向
        outputs = kn_model(mf, hidden, ps)
        hidden_detached = outputs["hidden"].detach()

        # 采样动作
        action_logits = outputs["action_logits"]
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()

        decision = {
            "action": int(action[0]),
            "position_size": float(outputs["position_size"].detach()[0, 0]),
            "sl_atr_mult": float(outputs["sl_atr_mult"].detach()[0, 0]),
            "tp_atr_mult": float(outputs["tp_atr_mult"].detach()[0, 0]),
            "confidence": float(outputs["confidence"].detach()[0, 0]),
            "should_trade": bool(outputs["should_trade_prob"].detach()[0, 0] > 0.5),
        }

        # 环境执行
        next_obs, reward, done, _, info = env.step(decision)

        log_prob = action_dist.log_prob(action[0])

        log_buffer.append({
            "log_prob": log_prob,
            "reward": reward,
            "action": decision["action"],
            "done": done,
            "hidden": hidden_detached,
        })

        episode_reward += reward
        episode_steps += 1

        if done or episode_steps >= sequence_length:
            # 简化的策略梯度
            returns = 0.0
            loss = torch.tensor(0.0)

            for entry in reversed(log_buffer):
                returns = entry["reward"] + 0.99 * returns
                loss = loss - entry["log_prob"] * returns

            if log_buffer:
                loss = loss / len(log_buffer)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(kn_model.parameters(), 0.5)
                optimizer.step()

            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_steps = 0
            log_buffer = []

            if done:
                obs, _ = env.reset()
                hidden = None

        hidden = hidden_detached
        total_rewards.append(reward)

        if step % log_interval == 0 and step > 0:
            avg_r = np.mean(total_rewards[-log_interval:])
            avg_ep = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {step:6d}: avg_reward={avg_r:.4f} avg_ep_reward={avg_ep:.4f}")

    return {
        "total_steps": total_timesteps,
        "avg_reward": float(np.mean(total_rewards)),
        "episode_rewards": [float(r) for r in episode_rewards[-20:]],
        "num_completed_episodes": len(episode_rewards),
    }


def _get_torch_local():
    from zhulong.agent.knowledge_net_kn2 import _ensure_torch
    return _ensure_torch()
