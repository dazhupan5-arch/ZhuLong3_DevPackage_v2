"""状态向量：struct(30) + embedding(32) + account/memory/time(12) + cognition(12) = 86。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from zhulong.agent.trader_memory import TraderMemory

REGIME_NAMES = [
    "trending_up",
    "trending_down",
    "ranging",
    "breakout_up",
    "breakout_down",
    "choppy",
    "unknown",
]
LEGACY_STATE_DIM = 74
COGNITION_DIM = 12  # calibrated_probs(3) + regime_onehot(7) + confidence + should_trade
KN2_COGNITION_DIM = 23  # 原12维 + action_onehot(6) + position_size + sl_mult + tp_mult + confidence + should_trade
STATE_DIM = LEGACY_STATE_DIM + COGNITION_DIM
KN2_STATE_DIM = LEGACY_STATE_DIM + KN2_COGNITION_DIM


def encode_regime_onehot(regime: str | None) -> np.ndarray:
    vec = np.zeros(len(REGIME_NAMES), dtype=np.float32)
    key = (regime or "unknown").lower()
    idx = REGIME_NAMES.index(key) if key in REGIME_NAMES else REGIME_NAMES.index("unknown")
    vec[idx] = 1.0
    return vec


def encode_cognition_features(
    calibrated_probs: np.ndarray | list[float] | None,
    regime: str | None,
    confidence: float,
    should_trade: bool,
) -> np.ndarray:
    p = np.asarray(calibrated_probs if calibrated_probs is not None else [], dtype=np.float32).reshape(-1)
    if p.size < 3:
        p = np.pad(p, (0, 3 - p.size))
    p = p[:3]
    tail = np.array([float(confidence), 1.0 if should_trade else 0.0], dtype=np.float32)
    return np.concatenate([p, encode_regime_onehot(regime), tail]).astype(np.float32)


def encode_kn2_cognition_features(
    kn2_decision: dict | None,
    calibrated_probs: np.ndarray | list[float] | None = None,
    regime: str | None = None,
    confidence: float = 0.0,
    should_trade: bool = False,
) -> np.ndarray:
    """编码 KN 2.0 认知特征为 23 维向量。

    包含: calibrated_probs(3) + regime_onehot(7) + confidence + should_trade
          + action_onehot(6) + position_size + sl_mult + tp_mult + confidence + should_trade
    """
    base = encode_cognition_features(calibrated_probs, regime, confidence, should_trade)

    if kn2_decision is None:
        # 无 KN 2.0 决策时补齐零
        tail = np.zeros(KN2_COGNITION_DIM - COGNITION_DIM, dtype=np.float32)
    else:
        action = int(kn2_decision.get("action", 0))
        action_onehot = np.zeros(6, dtype=np.float32)
        if 0 <= action < 6:
            action_onehot[action] = 1.0
        tail = np.array([
            *action_onehot,
            float(kn2_decision.get("position_size", 0.0)),
            float(kn2_decision.get("sl_atr_mult", 0.0)),
            float(kn2_decision.get("tp_atr_mult", 0.0)),
            float(kn2_decision.get("confidence", 0.0)),
            1.0 if kn2_decision.get("should_trade", False) else 0.0,
        ], dtype=np.float32)

    return np.concatenate([base, tail]).astype(np.float32)


def infer_regime_from_struct(struct_feat: np.ndarray) -> str:
    sf = np.asarray(struct_feat, dtype=np.float32).reshape(-1)
    trend = float(sf[0]) if sf.size else 0.0
    if trend > 0.65:
        return "breakout_up"
    if trend > 0.35:
        return "trending_up"
    if trend < -0.65:
        return "breakout_down"
    if trend < -0.35:
        return "trending_down"
    if abs(trend) < 0.12:
        return "ranging"
    return "choppy"


def primary_direction_from_probs(
    probs: np.ndarray | list[float] | None,
    regime: str | None = None,
    *,
    threshold: float = 0.42,
    regime_fallback: bool = True,
) -> str:
    """认知主方向：long / short / flat。标签顺序 0=空 1=平 2=多。"""
    p = np.asarray(probs if probs is not None else [], dtype=np.float32).reshape(-1)
    if p.size >= 3:
        short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])
        if long_p > short_p and long_p > flat_p and long_p >= threshold:
            return "long"
        if short_p > long_p and short_p > flat_p and short_p >= threshold:
            return "short"
    if not regime_fallback:
        return "flat"
    key = (regime or "").lower()
    if key in ("breakout_up", "trending_up"):
        return "long"
    if key in ("breakout_down", "trending_down"):
        return "short"
    return "flat"


class DirectionViewTracker:
    """
    跨 M5 tick 维持认知主观点（写入 agent_state，子进程每 tick 重载）。

    - 行情标签先做平滑，避免单根 M5 在 choppy/breakout 间抖动
    - 主方向需连续 min_hold_bars 根 M5 一致才翻转（从 flat 首次确立除外）
    """

    def __init__(
        self,
        *,
        min_hold_bars: int = 2,
        regime_min_hold_bars: int = 2,
        direction_threshold: float = 0.42,
        regime_fallback: bool = False,
        regime_min_confidence: float = 0.55,
    ) -> None:
        self.min_hold_bars = max(1, int(min_hold_bars))
        self.regime_min_hold_bars = max(1, int(regime_min_hold_bars))
        self.direction_threshold = float(direction_threshold)
        self.regime_fallback = bool(regime_fallback)
        self.regime_min_confidence = float(regime_min_confidence)
        self.primary_direction = "flat"
        self.bars_in_direction = 0
        self.smoothed_regime = "unknown"
        self._pending_direction = ""
        self._pending_direction_bars = 0
        self._pending_regime = ""
        self._pending_regime_bars = 0
        self._last_bar_key = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_direction": self.primary_direction,
            "bars_in_direction": self.bars_in_direction,
            "smoothed_regime": self.smoothed_regime,
            "pending_direction": self._pending_direction,
            "pending_direction_bars": self._pending_direction_bars,
            "pending_regime": self._pending_regime,
            "pending_regime_bars": self._pending_regime_bars,
            "last_bar_key": self._last_bar_key,
        }

    def load(self, blob: dict[str, Any] | None) -> None:
        if not blob:
            return
        self.primary_direction = str(blob.get("primary_direction") or "flat")
        self.bars_in_direction = int(blob.get("bars_in_direction") or 0)
        self.smoothed_regime = str(blob.get("smoothed_regime") or "unknown")
        self._pending_direction = str(blob.get("pending_direction") or "")
        self._pending_direction_bars = int(blob.get("pending_direction_bars") or 0)
        self._pending_regime = str(blob.get("pending_regime") or "")
        self._pending_regime_bars = int(blob.get("pending_regime_bars") or 0)
        self._last_bar_key = str(blob.get("last_bar_key") or "")

    def _smooth_regime(self, raw_regime: str, regime_conf: float) -> str:
        raw = (raw_regime or "unknown").lower()
        if self.smoothed_regime == "unknown":
            self.smoothed_regime = raw
            return self.smoothed_regime
        if raw == self.smoothed_regime:
            self._pending_regime = ""
            self._pending_regime_bars = 0
            return self.smoothed_regime
        if raw != self._pending_regime:
            self._pending_regime = raw
            self._pending_regime_bars = 1
        else:
            self._pending_regime_bars += 1
        need = self.regime_min_hold_bars
        if regime_conf >= 0.7:
            need = max(1, need - 1)
        if self._pending_regime_bars >= need:
            self.smoothed_regime = raw
            self._pending_regime = ""
            self._pending_regime_bars = 0
        return self.smoothed_regime

    def _instant_direction(
        self,
        calibrated_probs: np.ndarray | list[float],
        smoothed_regime: str,
        regime_conf: float,
    ) -> str:
        use_fallback = self.regime_fallback and regime_conf >= self.regime_min_confidence
        return primary_direction_from_probs(
            calibrated_probs,
            smoothed_regime if use_fallback else None,
            threshold=self.direction_threshold,
            regime_fallback=use_fallback,
        )

    def update(
        self,
        bar_key: str,
        calibrated_probs: np.ndarray | list[float],
        raw_regime: str,
        regime_conf: float,
    ) -> tuple[str, str, str]:
        """
        返回 (sticky_direction, smoothed_regime, instant_direction)。
        同一 bar_key 重复调用时直接返回当前观点（防御性）。
        """
        if bar_key and bar_key == self._last_bar_key:
            instant = self._instant_direction(calibrated_probs, self.smoothed_regime, regime_conf)
            return self.primary_direction, self.smoothed_regime, instant

        smoothed = self._smooth_regime(raw_regime, regime_conf)
        instant = self._instant_direction(calibrated_probs, smoothed, regime_conf)

        if bar_key:
            self._last_bar_key = bar_key

        if instant == "flat" or instant == self.primary_direction:
            self._pending_direction = ""
            self._pending_direction_bars = 0
            if self.primary_direction in ("long", "short") and instant == self.primary_direction:
                self.bars_in_direction += 1
            return self.primary_direction, smoothed, instant

        if self.primary_direction == "flat":
            self.primary_direction = instant
            self.bars_in_direction = 1
            self._pending_direction = ""
            self._pending_direction_bars = 0
            return self.primary_direction, smoothed, instant

        if instant != self._pending_direction:
            self._pending_direction = instant
            self._pending_direction_bars = 1
        else:
            self._pending_direction_bars += 1

        if self._pending_direction_bars >= self.min_hold_bars:
            self.primary_direction = instant
            self.bars_in_direction = 1
            self._pending_direction = ""
            self._pending_direction_bars = 0

        return self.primary_direction, smoothed, instant


def directional_confidence(
    probs: np.ndarray | list[float] | None,
    action_id: int,
    fallback: float,
) -> float:
    p = np.asarray(probs if probs is not None else [], dtype=np.float32).reshape(-1)
    if p.size >= 3:
        if action_id == 1:
            return float(p[2])
        if action_id in (2, 3, 4):
            return float(p[0])
    return float(fallback)


def action_entry_direction(action_id: int) -> str | None:
    if action_id == 1:
        return "long"
    if action_id in (2, 3, 4):
        return "short"
    return None


def gate_action_by_cognition(action_id: int, primary_dir: str) -> tuple[int, str]:
    """PPO 战术门控：禁止与认知主方向相反的入场。"""
    entry_dir = action_entry_direction(action_id)
    if entry_dir is None:
        return action_id, ""
    if primary_dir == "flat":
        return 0, "cognition_flat"
    if entry_dir != primary_dir:
        return 0, "cognition_direction_conflict"
    return action_id, ""


class StateBuilder:
    def __init__(self, scaler_path: str | Path | None = None) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if scaler_path and Path(scaler_path).is_file():
            data = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
            self.mean = np.array(data.get("mean", []), dtype=np.float32)
            self.std = np.array(data.get("std", []), dtype=np.float32)

    def save_scaler(self, features: np.ndarray, path: str | Path) -> None:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def _time_enc(self, ts) -> tuple[float, float, float, float]:
        if ts is None:
            return 0.0, 1.0, 0.0, 1.0
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        return (
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
        )

    def build(
        self,
        struct_feat: np.ndarray,
        embedding: np.ndarray,
        account: dict[str, Any],
        memory: TraderMemory,
        bar_time=None,
        cognition: dict[str, Any] | None = None,
    ) -> np.ndarray:
        struct_feat = np.asarray(struct_feat, dtype=np.float32).reshape(-1)[:30]
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)[:32]
        if struct_feat.size < 30:
            struct_feat = np.pad(struct_feat, (0, 30 - struct_feat.size))
        if embedding.size < 32:
            embedding = np.pad(embedding, (0, 32 - embedding.size))

        initial = float(account.get("initial_balance", 10000) or 10000)
        balance = float(account.get("balance", initial))
        equity = float(account.get("equity", balance))
        position = float(account.get("position", 0))
        peak = float(account.get("peak_balance", max(equity, initial)))
        daily_pnl = float(account.get("daily_pnl_r", 0))
        float_pnl = (equity - initial) / max(initial, 1e-9)
        drawdown = (peak - equity) / max(peak, 1e-9) if peak > 0 else 0.0
        cons_loss = float(memory.get_consecutive_losses())
        winrate = float(memory.get_winrate())
        avg_rr = float(memory.get_avg_rr())
        h_sin, h_cos, d_sin, d_cos = self._time_enc(bar_time)

        tail = np.array(
            [
                position,
                float_pnl,
                cons_loss,
                daily_pnl,
                peak / max(initial, 1e-9),
                drawdown,
                winrate,
                avg_rr,
                h_sin,
                h_cos,
                d_sin,
                d_cos,
            ],
            dtype=np.float32,
        )
        base = np.concatenate([struct_feat, embedding, tail])
        if base.shape[0] < LEGACY_STATE_DIM:
            base = np.pad(base, (0, LEGACY_STATE_DIM - base.shape[0]))
        base = base[:LEGACY_STATE_DIM]

        if cognition is None:
            regime = infer_regime_from_struct(struct_feat)
            cognition_tail = encode_cognition_features(None, regime, 0.0, False)
        else:
            cognition_tail = encode_cognition_features(
                cognition.get("calibrated_probs"),
                cognition.get("regime"),
                float(cognition.get("confidence") or 0.0),
                bool(cognition.get("should_trade")),
            )

        state = np.concatenate([base, cognition_tail])[:STATE_DIM]
        if state.shape[0] < STATE_DIM:
            state = np.pad(state, (0, STATE_DIM - state.shape[0]))

        scaler_dim = len(self.mean) if self.mean is not None else 0
        if self.mean is not None and self.std is not None and scaler_dim > 0:
            n = min(scaler_dim, STATE_DIM, len(self.std))
            state[:n] = (state[:n] - self.mean[:n]) / self.std[:n]
        np.nan_to_num(state, copy=False, nan=0.0, posinf=3.0, neginf=-3.0)
        return state.astype(np.float32)
