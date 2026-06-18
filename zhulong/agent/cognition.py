"""
交易员认知引擎（Trader Cognition Engine）。

在 KnowledgeNet + CausalInference 之上增加：
  1. 语境窗口（ContextWindow）— 多根 K 线上下文
  2. 市场状态识别（MarketRegime）— 趋势/震荡/突破
  3. 因果叙事追踪（CausalNarrative）— 因果变量时序追踪 + 卡尔曼滤波
  4. 信号交叉验证（SignalCrossValidator）— 多源信号一致性检查
  5. 信心校准（ConfidenceCalibrator）— 动态调整置信度
  6. 风险评估（RiskAssessor）— 动态仓位与止损
  7. 思维轨迹（ThoughtTrace）— 可读的推理日志

架构：
  StructureAnalyzer ──┐
  KnowledgeNet ───────┤
  CausalInference ────┼──→ CognitionEngine.process() ──→ ThoughtTrace
                       │        ↓
  ContextWindow ──────┘   adjusted_probs + confidence + regime + narrative
                            ↓
                       StateBuilder → PPO → action
"""

from __future__ import annotations

import json
import logging
import os
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 语境窗口：多根 K 线上下文
# ============================================================================

@dataclass
class BarContext:
    """单根 K 线的认知上下文。"""
    timestamp: str = ""
    struct_features: np.ndarray = field(default_factory=lambda: np.zeros(30, dtype=np.float32))
    knowledge_probs: np.ndarray = field(default_factory=lambda: np.array([0.34, 0.33, 0.33], dtype=np.float32))
    causal_pred: float = 0.0
    regime: str = "unknown"
    close: float = 0.0
    atr: float = 0.0
    volume: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp,
            "regime": self.regime,
            "causal_pred": round(self.causal_pred, 6),
            "probs": [round(float(x), 4) for x in self.knowledge_probs.reshape(-1)[:3]],
            "close": round(self.close, 4),
            "atr": round(self.atr, 4),
        }


class ContextWindow:
    """滚动窗口，维护最近 N 根 bar 的认知上下文。"""

    def __init__(self, max_bars: int = 36):
        self.max_bars = max_bars
        self.bars: deque[BarContext] = deque(maxlen=max_bars)

    def push(self, ctx: BarContext) -> None:
        self.bars.append(ctx)

    def latest(self) -> BarContext | None:
        return self.bars[-1] if self.bars else None

    def recent(self, n: int = 12) -> list[BarContext]:
        items = list(self.bars)
        return items[-n:] if len(items) >= n else items

    def struct_window(self, n: int = 12) -> np.ndarray:
        """获取最近 n 根 bar 的结构特征矩阵 (n, 30)。"""
        recent = self.recent(n)
        if not recent:
            return np.zeros((1, 30), dtype=np.float32)
        return np.stack([b.struct_features for b in recent])

    def prob_sequence(self, n: int = 12) -> np.ndarray:
        """最近 n 根 bar 的概率序列 (n, 3)。"""
        recent = self.recent(n)
        return np.stack([b.knowledge_probs[:3] for b in recent])

    def close_sequence(self, n: int = 12) -> np.ndarray:
        return np.array([b.close for b in self.recent(n)], dtype=np.float64)

    def volume_sequence(self, n: int = 12) -> np.ndarray:
        return np.array([b.volume for b in self.recent(n)], dtype=np.float64)

    def causal_sequence(self, n: int = 12) -> np.ndarray:
        return np.array([b.causal_pred for b in self.recent(n)], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.bars)

    def to_dict_list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b in self.bars:
            out.append(
                {
                    "timestamp": b.timestamp,
                    "regime": b.regime,
                    "causal_pred": b.causal_pred,
                    "close": b.close,
                    "atr": b.atr,
                    "volume": b.volume,
                    "struct_features": [float(x) for x in b.struct_features.reshape(-1)[:30]],
                    "knowledge_probs": [float(x) for x in b.knowledge_probs.reshape(-1)[:3]],
                }
            )
        return out

    def load_from_dict_list(self, items: list[dict[str, Any]] | None) -> None:
        self.bars.clear()
        if not items:
            return
        for item in items[-self.max_bars :]:
            sf = np.asarray(item.get("struct_features") or [], dtype=np.float32)
            if sf.size < 30:
                sf = np.pad(sf, (0, 30 - sf.size))[:30]
            kp = np.asarray(item.get("knowledge_probs") or [0.34, 0.33, 0.33], dtype=np.float32)
            if kp.size < 3:
                kp = np.array([0.34, 0.33, 0.33], dtype=np.float32)
            self.bars.append(
                BarContext(
                    timestamp=str(item.get("timestamp") or ""),
                    struct_features=sf,
                    knowledge_probs=kp[:3],
                    causal_pred=float(item.get("causal_pred") or 0.0),
                    regime=str(item.get("regime") or "unknown"),
                    close=float(item.get("close") or 0.0),
                    atr=float(item.get("atr") or 0.0),
                    volume=float(item.get("volume") or 0.0),
                )
            )


# ============================================================================
# 市场状态识别
# ============================================================================

class MarketRegime:
    """从最近 K 线识别市场状态。"""

    REGIMES = ["trending_up", "trending_down", "ranging", "breakout_up", "breakout_down", "choppy"]

    def __init__(self, atr_mult: float = 0.5):
        self.atr_mult = atr_mult
        self._current = "unknown"
        self._confidence = 0.0
        self._last_values: dict[str, float] = {}

    def detect(
        self,
        window: ContextWindow,
        struct: np.ndarray | None = None,
    ) -> tuple[str, float, dict[str, float]]:
        """
        检测当前市场状态。
        返回 (regime_name, confidence, metrics_dict)。
        """
        if len(window) < 6:
            self._current = "unknown"
            self._confidence = 0.0
            return self._current, self._confidence, {}

        closes = window.close_sequence(12)
        struct_recent = window.struct_window(12)

        # --- 趋势强度（从结构特征第 0 维：趋势分数） ---
        trend_scores = struct_recent[:, 0] if struct_recent.shape[1] > 0 else np.zeros(12)
        trend_mean = float(np.mean(trend_scores))
        trend_abs = abs(trend_mean)

        # --- 波动率 ---
        returns = np.diff(closes) / np.clip(closes[:-1], 1e-9, None)
        vol = float(np.std(returns)) if len(returns) > 0 else 0.0

        # --- 价格位置（在近期范围内的位置） ---
        high = np.max(closes)
        low = np.min(closes)
        price_range = high - low
        pos_in_range = (closes[-1] - low) / max(price_range, 1e-9)  # 0~1

        # --- 突破检测 ---
        if len(closes) >= 8:
            prev_high = np.max(closes[:6])
            prev_low = np.min(closes[:6])
            breakout_up = closes[-1] > prev_high * 1.005
            breakout_down = closes[-1] < prev_low * 0.995
        else:
            breakout_up = False
            breakout_down = False

        # --- 分类 ---
        if breakout_up and trend_mean > 0:
            regime = "breakout_up"
            conf = min(0.85, 0.6 + trend_abs * 2.0)
        elif breakout_down and trend_mean < 0:
            regime = "breakout_down"
            conf = min(0.85, 0.6 + trend_abs * 2.0)
        elif trend_abs > 0.08:
            regime = "trending_up" if trend_mean > 0 else "trending_down"
            conf = min(0.9, 0.55 + trend_abs * 3.0)
        elif vol < 0.0005 and price_range / closes[-1] < 0.005:
            regime = "ranging"
            conf = min(0.8, 0.5 + (1.0 - vol * 500))
        else:
            regime = "choppy"
            conf = 0.35

        metrics = {
            "trend_mean": round(trend_mean, 6),
            "vol": round(vol, 6),
            "pos_in_range": round(pos_in_range, 4),
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "price_range_pct": round(price_range / max(closes[-1], 1e-9) * 100, 3),
        }

        self._current = regime
        self._confidence = conf
        self._last_values = metrics
        return regime, conf, metrics

    @property
    def current(self) -> str:
        return self._current

    @property
    def confidence(self) -> float:
        return self._confidence

    def is_trending(self) -> bool:
        return self._current in ("trending_up", "trending_down", "breakout_up", "breakout_down")

    def is_ranging(self) -> bool:
        return self._current == "ranging"

    def trend_direction(self) -> int:
        """1=上升趋势, -1=下降趋势, 0=无明确方向。"""
        if self._current in ("trending_up", "breakout_up"):
            return 1
        if self._current in ("trending_down", "breakout_down"):
            return -1
        return 0


# ============================================================================
# 因果叙事追踪（卡尔曼滤波 + 时序状态）
# ============================================================================

@dataclass
class CausalState:
    """一个时间点的因果变量状态。"""
    macro_shock: float = 0.0
    risk_aversion: float = 0.0
    dollar_index: float = 0.0
    demand: float = 0.0
    price_change: float = 0.0
    timestamp: str = ""


class CausalNarrative:
    """
    追踪因果变量随时间的变化，构建"叙事"。

    用简单的一阶指数平滑（EMA）追踪每个因果变量的状态，
    检测状态突变（shift detection），生成因果叙事文本。
    """

    def __init__(self, symbol: str = "XAUUSD", alpha: float = 0.2):
        self.symbol = symbol
        self.alpha = alpha  # EMA 平滑系数
        self._state = CausalState()
        self._prev_state = CausalState()
        self._history: deque[CausalState] = deque(maxlen=20)
        self._shift_count = 0

    def update(
        self,
        macro_shock: float,
        risk_aversion: float,
        dollar_index: float,
        demand: float,
        price_change: float,
        timestamp: str = "",
    ) -> tuple[CausalState, list[str]]:
        """更新因果状态，返回 (当前状态, 叙事事件列表)。"""
        self._prev_state = CausalState(
            macro_shock=self._state.macro_shock,
            risk_aversion=self._state.risk_aversion,
            dollar_index=self._state.dollar_index,
            demand=self._state.demand,
            price_change=self._state.price_change,
            timestamp=self._state.timestamp,
        )

        # EMA 平滑
        a = self.alpha
        s = self._state
        s.macro_shock = s.macro_shock * (1 - a) + macro_shock * a
        s.risk_aversion = s.risk_aversion * (1 - a) + risk_aversion * a
        s.dollar_index = s.dollar_index * (1 - a) + dollar_index * a
        s.demand = s.demand * (1 - a) + demand * a
        s.price_change = s.price_change * (1 - a) + price_change * a
        s.timestamp = timestamp

        self._history.append(CausalState(
            macro_shock=s.macro_shock,
            risk_aversion=s.risk_aversion,
            dollar_index=s.dollar_index,
            demand=s.demand,
            price_change=s.price_change,
            timestamp=timestamp,
        ))

        # 检测叙事事件
        events = self._detect_narrative_events()

        # 检测状态突变
        if self._detect_shift():
            self._shift_count += 1
            events.append(f"因果状态突变 #{self._shift_count}")

        return s, events

    def _detect_shift(self, threshold: float = 1.5) -> bool:
        """检测因果状态是否发生显著突变。"""
        d_macro = abs(self._state.macro_shock - self._prev_state.macro_shock)
        d_risk = abs(self._state.risk_aversion - self._prev_state.risk_aversion)
        d_dollar = abs(self._state.dollar_index - self._prev_state.dollar_index)
        total_shift = d_macro + d_risk + d_dollar
        return total_shift > threshold

    def _detect_narrative_events(self) -> list[str]:
        events: list[str] = []
        s = self._state

        if s.macro_shock > 0.5:
            events.append("宏观冲击偏正（风险偏好上升）")
        elif s.macro_shock < -0.5:
            events.append("宏观冲击偏负（避险情绪上升）")

        if s.risk_aversion > 0.3:
            events.append("风险厌恶上升（资金流入避险资产）")
        elif s.risk_aversion < -0.3:
            events.append("风险偏好上升（资金流出避险资产）")

        if s.dollar_index > 0.2:
            event = "美元走强（压制" + ("黄金" if self.symbol == "XAUUSD" else "原油") + "）"
            events.append(event)
        elif s.dollar_index < -0.2:
            event = "美元走弱（支撑" + ("黄金" if self.symbol == "XAUUSD" else "原油") + "）"
            events.append(event)

        if s.demand > 0.2:
            events.append("需求端走强")
        elif s.demand < -0.2:
            events.append("需求端走弱")

        if s.price_change > 0.002:
            events.append("因果模型预测价格上涨")
        elif s.price_change < -0.002:
            events.append("因果模型预测价格下跌")

        return events

    @property
    def state(self) -> CausalState:
        return self._state

    def narrative_summary(self) -> str:
        """生成一句话因果叙事。"""
        s = self._state
        parts = []
        if s.macro_shock > 0.2:
            parts.append("宏观环境偏积极")
        elif s.macro_shock < -0.2:
            parts.append("宏观环境偏消极")
        else:
            parts.append("宏观环境中性")

        if s.demand > 0.1:
            parts.append("需求走强")
        elif s.demand < -0.1:
            parts.append("需求走弱")

        if s.price_change > 0.001:
            parts.append("→ 偏多")
        elif s.price_change < -0.001:
            parts.append("→ 偏空")
        else:
            parts.append("→ 观望")

        return " | ".join(parts)

    def simulate_intervention(self, intervention: str, value: float) -> CausalState:
        """
        反事实模拟：如果因果变量 X 被干预为 value，结果会怎样？
        支持的干预变量: macro_shock, dollar_index, risk_aversion.

        返回模拟后的 CausalState。
        """
        s = self._state
        sim = CausalState(
            macro_shock=s.macro_shock,
            risk_aversion=s.risk_aversion,
            dollar_index=s.dollar_index,
            demand=s.demand,
            price_change=s.price_change,
        )

        if intervention == "macro_shock":
            sim.macro_shock = value
        elif intervention == "dollar_index":
            sim.dollar_index = value
        elif intervention == "risk_aversion":
            sim.risk_aversion = value
        else:
            return sim

        # 重新计算下游
        # 简化：风险厌恶受 macro_shock 影响
        sim.risk_aversion = sim.risk_aversion * 0.7 + sim.macro_shock * 0.3
        # 美元受 macro_shock 影响
        sim.dollar_index = sim.dollar_index * 0.7 + sim.macro_shock * 0.3
        # 需求受风险厌恶和美元综合影响
        demand_cfg = self.symbol
        sim.demand = sim.demand * 0.5 + (sim.risk_aversion * -0.3 + sim.dollar_index * -0.2)
        # 价格变化
        sim.price_change = sim.demand * 0.002

        return sim

    def intervention_analysis(self) -> dict[str, Any]:
        """
        生成反事实分析报告：如果关键变量变化，价格会怎样？
        """
        current = self._state
        analysis = {
            "current": {
                "price_change": round(current.price_change, 6),
                "direction": "看多" if current.price_change > 0.001 else ("看空" if current.price_change < -0.001 else "中性"),
            },
            "scenarios": {},
        }

        # 情景 1：宏观冲击 +1σ
        sim_up = self.simulate_intervention("macro_shock", current.macro_shock + 1.0)
        analysis["scenarios"]["macro_positive_shock"] = {
            "description": "如果宏观环境突然改善",
            "simulated_price_change": round(sim_up.price_change, 6),
            "delta": round(sim_up.price_change - current.price_change, 6),
        }

        # 情景 2：宏观冲击 -1σ
        sim_down = self.simulate_intervention("macro_shock", current.macro_shock - 1.0)
        analysis["scenarios"]["macro_negative_shock"] = {
            "description": "如果宏观环境突然恶化",
            "simulated_price_change": round(sim_down.price_change, 6),
            "delta": round(sim_down.price_change - current.price_change, 6),
        }

        # 情景 3：美元走强
        sim_dollar = self.simulate_intervention("dollar_index", 1.0)
        analysis["scenarios"]["dollar_strengthens"] = {
            "description": "如果美元大幅走强",
            "simulated_price_change": round(sim_dollar.price_change, 6),
            "delta": round(sim_dollar.price_change - current.price_change, 6),
        }

        return analysis

    def history_dicts(self) -> list[dict]:
        return [
            {
                "ts": h.timestamp,
                "macro_shock": round(h.macro_shock, 4),
                "risk_aversion": round(h.risk_aversion, 4),
                "dollar_index": round(h.dollar_index, 4),
                "demand": round(h.demand, 4),
                "price_change": round(h.price_change, 6),
            }
            for h in self._history
        ]


# ============================================================================
# 信号交叉验证
# ============================================================================

class SignalCrossValidator:
    """
    检查多个信号源的一致性：
      - 结构分析（trend direction）
      - 知识网络（probability）
      - 因果推理（price_change prediction）
      - 市场状态（regime）
    返回一致性得分和冲突详情。
    """

    def validate(
        self,
        knowledge_probs: np.ndarray,
        causal_pred: float,
        regime: MarketRegime,
        struct: np.ndarray,
    ) -> tuple[float, list[str], dict[str, float]]:
        """
        返回 (agreement_score 0~1, conflicts, signal_strengths)。
        """
        conflicts: list[str] = []
        signals: dict[str, float] = {}

        p = knowledge_probs.reshape(-1)[:3]
        # 与 KnowledgeNet 训练标签一致：0=空 1=观望 2=多
        short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])

        # --- 1. 知识网络信号 ---
        kn_direction = 0
        kn_strength = flat_p
        if long_p > flat_p and long_p > short_p:
            kn_direction = 1
            kn_strength = long_p - max(flat_p, short_p)
        elif short_p > flat_p and short_p > long_p:
            kn_direction = -1
            kn_strength = short_p - max(flat_p, long_p)
        signals["knowledge_direction"] = float(kn_direction)
        signals["knowledge_strength"] = round(kn_strength, 4)

        # --- 2. 因果信号 ---
        causal_direction = 1 if causal_pred > 0.001 else (-1 if causal_pred < -0.001 else 0)
        causal_strength = abs(causal_pred) / 0.003  # 归一化
        signals["causal_direction"] = float(causal_direction)
        signals["causal_strength"] = round(min(causal_strength, 1.0), 4)

        # --- 3. 结构信号 ---
        trend_dir = regime.trend_direction()
        trend_conf = regime.confidence
        signals["structure_direction"] = float(trend_dir)
        signals["structure_confidence"] = round(trend_conf, 4)

        # --- 4. 市场状态信号 ---
        regime_is_trending = 1.0 if regime.is_trending() else 0.0
        regime_is_ranging = 1.0 if regime.is_ranging() else 0.0
        signals["regime_trending"] = regime_is_trending
        signals["regime_ranging"] = regime_is_ranging

        # --- 一致性检查 ---
        directional_signals = []
        if kn_direction != 0:
            directional_signals.append(("知识网络", kn_direction))
        if causal_direction != 0:
            directional_signals.append(("因果推理", causal_direction))
        if trend_dir != 0:
            directional_signals.append(("市场结构", trend_dir))

        agreement_count = 0
        total_directional = len(directional_signals)

        if total_directional >= 2:
            # 检查所有方向信号是否一致
            first_dir = directional_signals[0][1]
            all_agree = all(d[1] == first_dir for d in directional_signals)
            if all_agree:
                agreement_count = total_directional
            else:
                for i in range(total_directional):
                    for j in range(i + 1, total_directional):
                        if directional_signals[i][1] != directional_signals[j][1]:
                            conflicts.append(
                                f"{directional_signals[i][0]} vs {directional_signals[j][0]} 方向冲突"
                            )
                agreement_count = 0

        # --- 市场状态一致性 ---
        if regime_is_ranging and kn_direction != 0:
            conflicts.append("震荡市发出方向信号 → 降低置信度")
        if not regime_is_trending and not regime_is_ranging and kn_direction != 0:
            conflicts.append("市场方向不明 → 谨慎交易")

        # --- 计算一致性得分 ---
        if total_directional == 0:
            agreement_score = 0.3  # 无方向信号，观望
        elif total_directional == 1:
            agreement_score = 0.5  # 仅一个信号
        elif agreement_count == total_directional:
            agreement_score = 0.85 + 0.15 * (total_directional - 2) / 2  # 全部一致
        else:
            agreement_score = 0.3 + 0.3 * (agreement_count / total_directional)  # 部分一致

        # 状态加成
        if regime_is_trending and agreement_score > 0.5:
            agreement_score = min(1.0, agreement_score + 0.1)

        signals["agreement_score"] = round(agreement_score, 4)
        return agreement_score, conflicts, signals


# ============================================================================
# 信心校准
# ============================================================================

class ConfidenceCalibrator:
    """根据多源信号一致性和市场状态动态调整置信度。"""

    def __init__(self, base_threshold: float = 0.60):
        self.base_threshold = base_threshold
        self._recent_accuracy: deque[bool] = deque(maxlen=20)

    def calibrate(
        self,
        knowledge_probs: np.ndarray,
        agreement_score: float,
        regime: MarketRegime,
        conflicts: list[str],
        causal_pred: float,
        lock_direction: str | None = None,
    ) -> tuple[np.ndarray, float, str]:
        """
        返回 (calibrated_probs, confidence, reasoning_chain)。
        """
        p = knowledge_probs.reshape(-1)[:3].copy()
        # 与 KnowledgeNet 训练标签一致：0=空 1=观望 2=多
        short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])

        if lock_direction in ("long", "short", "flat"):
            reasoning = f"V16 锁定方向={lock_direction}（不改写预测概率）"
            conf = max(short_p, flat_p, long_p)
            return p.astype(np.float32), conf, reasoning

        reasoning_steps: list[str] = []

        # --- Step 1: 基础信号 ---
        max_prob = max(long_p, short_p, flat_p)
        if long_p > short_p and long_p > flat_p:
            reasoning_steps.append(f"知识网络看多 (概率 {long_p:.1%})")
            base_signal = "long"
            base_conf = long_p
        elif short_p > long_p and short_p > flat_p:
            reasoning_steps.append(f"知识网络看空 (概率 {short_p:.1%})")
            base_signal = "short"
            base_conf = short_p
        else:
            reasoning_steps.append(f"知识网络观望 (flat {flat_p:.1%})")
            base_signal = "flat"
            base_conf = flat_p

        # --- Step 2: 因果推理 ---
        if abs(causal_pred) > 0.001:
            causal_dir = "多" if causal_pred > 0 else "空"
            reasoning_steps.append(f"因果推理偏{causal_dir} (预测 {causal_pred:.4%})")
        else:
            reasoning_steps.append("因果推理中性")

        # --- Step 3: 市场状态 ---
        reasoning_steps.append(f"市场状态: {regime.current} (置信度 {regime.confidence:.1%})")

        # --- Step 4: 信号一致性 ---
        if conflicts:
            for c in conflicts:
                reasoning_steps.append(f"冲突: {c}")
        else:
            reasoning_steps.append("多源信号一致")

        # --- Step 5: 校准 ---
        calibrated = p.copy()
        confidence = base_conf

        # 一致性越高，方向信号越可信
        if base_signal != "flat":
            if agreement_score > 0.8:
                # 高一致性：增强方向信号
                boost = (agreement_score - 0.5) * 0.3
                if base_signal == "long":
                    calibrated[2] += boost
                    calibrated[0] -= boost * 0.5
                    calibrated[1] -= boost * 0.5
                else:
                    calibrated[0] += boost
                    calibrated[1] -= boost * 0.5
                    calibrated[2] -= boost * 0.5
                reasoning_steps.append(f"高一致性 (+{boost:.2f}) → 强化方向信号")
            elif agreement_score < 0.4 and conflicts:
                # 低一致性有冲突：削弱方向，增强观望
                decay = 0.3
                if base_signal == "long":
                    calibrated[2] -= decay
                    calibrated[1] += decay
                else:
                    calibrated[0] -= decay
                    calibrated[1] += decay
                reasoning_steps.append(f"信号冲突 (-{decay:.2f}) → 偏向观望")

        # 震荡市压制方向信号
        if regime.is_ranging() and base_signal != "flat":
            calibrated[1] += 0.15
            if base_signal == "long":
                calibrated[2] -= 0.15
            else:
                calibrated[0] -= 0.15
            reasoning_steps.append("震荡市压制 → 增加观望权重")

        # 归一化
        calibrated = np.clip(calibrated, 0.01, None)
        calibrated /= calibrated.sum()

        # 最终置信度
        confidence = float(max(calibrated))
        reasoning_steps.append(f"校准后置信度: {confidence:.1%}")

        # --- 最近准确率调整 ---
        if len(self._recent_accuracy) >= 5:
            recent_acc = sum(self._recent_accuracy) / len(self._recent_accuracy)
            if recent_acc < 0.4:
                confidence *= 0.85
                reasoning_steps.append(f"近期准确率低 ({recent_acc:.0%}) → 降置信度")

        reasoning_chain = " → ".join(reasoning_steps)
        return calibrated.astype(np.float32), confidence, reasoning_chain

    def record_outcome(self, was_correct: bool) -> None:
        self._recent_accuracy.append(was_correct)


# ============================================================================
# 风险评估
# ============================================================================

@dataclass
class RiskAssessment:
    """风险评估结果。"""
    risk_score: float = 0.5         # 0=低风险, 1=高风险
    position_mult: float = 1.0      # 仓位倍率
    sl_mult: float = 1.2           # 止损 ATR 倍率
    tp_mult: float = 2.0           # 止盈 ATR 倍率
    should_trade: bool = True
    warnings: list[str] = field(default_factory=list)
    # ===== P2-2: AI 计算的具体止损/止盈价格 =====
    sl_price: float = 0.0           # 智能体计算的止损价格
    tp_price: float = 0.0           # 智能体计算的止盈价格
    sl_reasoning: str = ""          # 止损计算理由
    # ===== 结束 =====


class RiskAssessor:
    """基于波动率、市场状态、近期表现评估风险。"""

    def __init__(self, max_position_mult: float = 1.2, min_position_mult: float = 0.2):
        self.max_position_mult = max_position_mult
        self.min_position_mult = min_position_mult

    def assess(
        self,
        regime: MarketRegime,
        confidence: float,
        vol: float,
        consecutive_losses: int,
        time_of_day: tuple[float, float] | None = None,
        # ===== P2-2: 新增参数用于市场结构 SL/TP =====
        struct_features: np.ndarray | None = None,
        close: float | None = None,
        atr: float | None = None,
        trade_bias: str = "long",
        entry_anchored: bool = False,
        # ===== 结束 =====
        atr_mean: float = 0.0,
    ) -> RiskAssessment:
        warnings: list[str] = []
        risk_score = 0.5

        # --- 市场状态风险 ---
        if regime.current == "choppy":
            risk_score += 0.2
            warnings.append("无序波动 → 风险上升")
        elif regime.is_ranging():
            risk_score += 0.05
            warnings.append("震荡 → 轻仓")
        elif regime.is_trending():
            risk_score -= 0.1
            warnings.append("趋势明确 → 风险可控")

        # --- 波动率风险 ---
        if vol > 0.003:
            risk_score += 0.15
            warnings.append("高波动 → 加宽止损")
        elif vol < 0.0008:
            risk_score -= 0.05

        # --- 连亏风险 ---
        if consecutive_losses >= 3:
            risk_score += 0.15
            warnings.append(f"连亏{consecutive_losses}笔 → 减仓")
        if consecutive_losses >= 5:
            risk_score += 0.1
            warnings.append("严重连亏 → 暂停交易")

        # --- 时间风险（亚洲盘/午盘通常波动小） ---
        if time_of_day:
            hour = time_of_day[0]
            if 1 <= hour < 7:  # 亚洲早盘
                risk_score += 0.05
                warnings.append("亚洲早盘流动性低")

        # --- 开仓前 ATR 异常放大检查 ---
        if atr_mean > 0 and atr is not None and atr > 0 and atr > atr_mean * 1.8:
            risk_score += 0.2
            warnings.append("ATR异常放大 → 危险入场")

        risk_score = np.clip(risk_score, 0.0, 1.0)

        # --- 仓位倍率（风险越高仓位越小） ---
        position_mult = 1.0 - (risk_score - 0.5) * 1.5
        position_mult = np.clip(position_mult, self.min_position_mult, self.max_position_mult)

        # --- 止损倍率 ---
        sl_mult = 1.2 + (risk_score - 0.5) * 1.5
        sl_mult = np.clip(sl_mult, 1.0, 2.5)
        tp_mult = sl_mult * 1.67  # RR ≈ 1.67

        # ===== P2-2: 基于市场结构的自适应 SL/TP 价格 =====
        sl_price = 0.0
        tp_price = 0.0
        sl_reasoning = ""

        bias = (trade_bias or "long").strip().lower()
        if bias not in ("long", "short"):
            bias = "long"

        if close is not None and atr is not None and atr > 0:
            if entry_anchored:
                if bias == "long":
                    sl_price = close - atr * sl_mult
                    tp_price = close + atr * tp_mult
                    resistance_level = self._extract_resistance(struct_features, close, atr)
                    if resistance_level > close and resistance_level < tp_price:
                        tp_price = max(
                            resistance_level + atr * 0.25,
                            close + atr * sl_mult * 1.5,
                        )
                    sl_reasoning = f"多SL相对入场ATR×{sl_mult:.1f}"
                else:
                    sl_price = close + atr * sl_mult
                    tp_price = close - atr * tp_mult
                    support_level = self._extract_support(struct_features, close, atr)
                    if support_level > 0 and support_level > tp_price and support_level < close:
                        tp_price = min(
                            support_level - atr * 0.25,
                            close - atr * sl_mult * 1.5,
                        )
                    sl_reasoning = f"空SL相对入场ATR×{sl_mult:.1f}"
            else:
                support_level = self._extract_support(struct_features, close, atr)
                resistance_level = self._extract_resistance(struct_features, close, atr)
                buffer = atr * 0.3

                if bias == "short":
                    if resistance_level > 0 and close > 0:
                        computed_sl = resistance_level + buffer
                        atr_sl = close + atr * sl_mult
                        sl_price = max(computed_sl, atr_sl)
                        sl_reasoning = f"空SL基于阻力位{resistance_level:.2f}"
                        warnings.append(f"空SL基于阻力位{resistance_level:.2f}")
                    else:
                        sl_price = close + atr * sl_mult
                        sl_reasoning = f"空ATR倍率{sl_mult:.1f}"

                    if support_level > 0 and close > 0:
                        computed_tp = support_level - atr * 0.5
                        atr_tp = close - atr * tp_mult
                        tp_price = min(computed_tp, atr_tp) if computed_tp > 0 else atr_tp
                        warnings.append(f"空TP基于支撑位{support_level:.2f}")
                    else:
                        tp_price = close - atr * tp_mult
                else:
                    if support_level > 0 and close > 0:
                        computed_sl = support_level - buffer
                        atr_sl = close - atr * sl_mult
                        sl_price = min(computed_sl, atr_sl) if computed_sl > 0 else atr_sl
                        sl_reasoning = f"多SL基于支撑位{support_level:.2f}"
                        warnings.append(f"SL基于支撑位{support_level:.2f}")
                    else:
                        sl_price = close - atr * sl_mult
                        sl_reasoning = f"多ATR倍率{sl_mult:.1f}"

                    if resistance_level > 0 and close > 0:
                        computed_tp = resistance_level + atr * 0.5
                        atr_tp = close + atr * tp_mult
                        tp_price = max(computed_tp, atr_tp * 0.8)
                        warnings.append(f"TP基于阻力位{resistance_level:.2f}")
                    else:
                        tp_price = close + atr * tp_mult

        if close is not None and atr is not None and atr > 0:
            if sl_price <= 0:
                sl_price = (close + atr * sl_mult) if bias == "short" else (close - atr * sl_mult)
            if tp_price <= 0:
                tp_price = (close - atr * tp_mult) if bias == "short" else (close + atr * tp_mult)
        # ===== 结束 =====

        # --- 是否交易 ---
        should_trade = True
        if confidence < 0.55:
            should_trade = False
            warnings.append("置信度过低 → 不入场")
        if risk_score > 0.8:
            should_trade = False
            warnings.append("风险过高 → 不入场")
        # --- Choppy + 高波动禁止入场 ---
        if regime.current == "choppy" and vol > 0.003:
            should_trade = False
            warnings.append("无序波动+高波动 → 不入场")
        if consecutive_losses >= 6:
            should_trade = False
            warnings.append("过度连亏 → 暂停")

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            position_mult=round(position_mult, 4),
            sl_mult=round(sl_mult, 3),
            tp_mult=round(tp_mult, 3),
            should_trade=should_trade,
            warnings=warnings,
            sl_price=round(sl_price, 5) if sl_price else 0.0,
            tp_price=round(tp_price, 5) if tp_price else 0.0,
            sl_reasoning=sl_reasoning,
        )

    def _extract_support(self, struct: np.ndarray | None, close: float, atr: float) -> float:
        """从 struct[3]=m5_support_dist 还原支撑位价格：sup = close - dist×ATR。"""
        if struct is None or struct.size < 4 or close <= 0 or atr <= 0:
            return 0.0
        try:
            dist = float(struct[3])
            if not np.isfinite(dist) or dist <= 0:
                return 0.0
            sup = close - dist * atr
            if sup > 0 and close - sup > 5 * atr:
                return 0.0
            return sup if sup > 0 else 0.0
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _extract_resistance(self, struct: np.ndarray | None, close: float, atr: float) -> float:
        """从 struct[4]=m5_resistance_dist 还原阻力位价格：res = close + dist×ATR。"""
        if struct is None or struct.size < 5 or close <= 0 or atr <= 0:
            return 0.0
        try:
            dist = float(struct[4])
            if not np.isfinite(dist) or dist <= 0:
                return 0.0
            res = close + dist * atr
            if res > 0 and res - close > 5 * atr:
                return 0.0
            return res if res > 0 else 0.0
        except (IndexError, TypeError, ValueError):
            return 0.0


# ============================================================================
# 思维轨迹
# ============================================================================

@dataclass
class ThoughtTrace:
    """单次推理的完整思维轨迹。"""
    timestamp: str = ""
    symbol: str = ""
    # 输入
    knowledge_probs: list[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    causal_pred: float = 0.0
    # 认知层
    regime: str = "unknown"
    regime_confidence: float = 0.0
    narrative: str = ""
    narrative_events: list[str] = field(default_factory=list)
    agreement_score: float = 0.0
    conflicts: list[str] = field(default_factory=list)
    # 反事实分析
    intervention_analysis: dict[str, Any] = field(default_factory=dict)
    # 输出
    calibrated_probs: list[float] = field(default_factory=lambda: [0.33, 0.34, 0.33])
    confidence: float = 0.0
    reasoning_chain: str = ""
    # 风险
    risk_score: float = 0.5
    risk_warnings: list[str] = field(default_factory=list)
    position_mult: float = 1.0
    should_trade: bool = True
    sl_mult: float = 1.2
    tp_mult: float = 2.0
    trade_bias: str = "long"
    # ===== P2-2: AI 平仓评估 =====
    exit_assessment: float = 0.0    # 0=继续持有, 1=建议平仓
    exit_reason: str = ""
    ai_sl_price: float = 0.0        # 智能体建议的止损价
    ai_tp_price: float = 0.0        # 智能体建议的止盈价
    ai_entry_price: float = 0.0      # 智能体 tick 评估的理想入场价
    entry_should_wait: bool = False  # 是否应等待更优价（追价过高）
    entry_wait_reason: str = ""
    exit_reasoning: str = ""        # 平仓/调整理由
    trail_mode: str = "hold"        # hold | run | tighten
    suggested_trailing_sl: float = 0.0
    position_mgmt_reason: str = ""
    # ===== 结束 =====

    def to_dict(self) -> dict[str, Any]:
        def _rf(v: float, n: int = 4) -> float:
            x = float(v)
            return round(x, n) if math.isfinite(x) else 0.0

        return {
            "ts": self.timestamp,
            "symbol": self.symbol,
            "knowledge_probs": [_rf(x) for x in self.knowledge_probs],
            "causal_pred": _rf(self.causal_pred, 6),
            "regime": self.regime,
            "regime_confidence": _rf(self.regime_confidence),
            "narrative": self.narrative,
            "narrative_events": self.narrative_events,
            "intervention_analysis": self.intervention_analysis,
            "agreement_score": _rf(self.agreement_score),
            "conflicts": self.conflicts,
            "calibrated_probs": [_rf(x) for x in self.calibrated_probs],
            "confidence": _rf(self.confidence),
            "reasoning_chain": self.reasoning_chain,
            "risk_score": _rf(self.risk_score),
            "risk_warnings": self.risk_warnings,
            "position_mult": _rf(self.position_mult),
            "should_trade": self.should_trade,
            # ===== P2-2 =====
            "exit_assessment": _rf(self.exit_assessment),
            "exit_reason": self.exit_reason,
            "ai_sl_price": round(self.ai_sl_price, 5) if self.ai_sl_price else 0.0,
            "ai_tp_price": round(self.ai_tp_price, 5) if self.ai_tp_price else 0.0,
            "ai_entry_price": round(self.ai_entry_price, 5) if self.ai_entry_price else 0.0,
            "entry_should_wait": self.entry_should_wait,
            "entry_wait_reason": self.entry_wait_reason,
            "exit_reasoning": self.exit_reasoning,
            "trail_mode": self.trail_mode,
            "suggested_trailing_sl": round(self.suggested_trailing_sl, 5)
            if self.suggested_trailing_sl
            else 0.0,
            "position_mgmt_reason": self.position_mgmt_reason,
            # ===== 结束 =====
        }

    def to_log_line(self) -> str:
        """单行摘要日志。"""
        direction = (
            "多" if self.calibrated_probs[2] > max(self.calibrated_probs[0], self.calibrated_probs[1])
            else "空" if self.calibrated_probs[0] > max(self.calibrated_probs[1], self.calibrated_probs[2])
            else "观望"
        )
        trade = "✓可交易" if self.should_trade else "✗不交易"
        conflicts_str = f" [{', '.join(self.conflicts)}]" if self.conflicts else ""
        # ===== P2-2: 追加平仓评估信息 =====
        exit_str = ""
        if self.exit_assessment > 0.5:
            exit_str = f" | 平仓建议({self.exit_reason})"
        elif self.ai_sl_price > 0 or self.ai_tp_price > 0:
            exit_str = f" | AI止损={self.ai_sl_price:.2f} AI止盈={self.ai_tp_price:.2f}"
        # ===== 结束 =====
        return (
            f"[{self.timestamp}] {self.symbol} | 状态={self.regime} | "
            f"置信={self.confidence:.1%} | 方向={direction} | {trade}"
            f"{conflicts_str} | 仓位×{self.position_mult:.2f}{exit_str}"
        )


# ============================================================================
# 认知引擎（主入口）
# ============================================================================

class CognitionEngine:
    """
    交易员认知引擎。

    用法:
        engine = CognitionEngine(config)
        # 每根新 bar:
        thought = engine.process(
            struct_features, knowledge_probs, causal_pred,
            close, atr, volume, bar_timestamp, consecutive_losses
        )
        # thought.calibrated_probs → 输入 PPO
        # thought.to_log_line() → 思维日志
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        cc = cfg.get("cognition", {})
        self.enabled = bool(cc.get("enabled", True))
        self.symbol = str(cc.get("symbol", "XAUUSD")).upper()

        # 子模块
        self.context = ContextWindow(max_bars=int(cc.get("context_bars", 36)))
        self.regime = MarketRegime(atr_mult=float(cc.get("regime_atr_mult", 0.5)))
        self.narrative = CausalNarrative(
            symbol=self.symbol,
            alpha=float(cc.get("narrative_alpha", 0.2)),
        )
        self.validator = SignalCrossValidator()
        self.calibrator = ConfidenceCalibrator(
            base_threshold=float(cc.get("base_confidence_threshold", 0.60)),
        )
        self.risk = RiskAssessor(
            max_position_mult=float(cc.get("max_position_mult", 1.2)),
            min_position_mult=float(cc.get("min_position_mult", 0.2)),
        )

        # 日志（安装目录只读，写入 AppData/logs）
        from zhulong.utils.paths import resolve_writable_log_path

        log_dir = resolve_writable_log_path(cc.get("thought_log_dir", "logs/cognition"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._thought_log_path = log_dir / f"thought_trace_{self.symbol}.jsonl"

        self._bar_count = 0
        from zhulong.agent.state_builder import DirectionViewTracker

        self._direction_view = DirectionViewTracker(
            min_hold_bars=int(cc.get("direction_min_hold_bars", 2)),
            regime_min_hold_bars=int(cc.get("regime_min_hold_bars", 2)),
            direction_threshold=float(cc.get("direction_threshold", 0.42)),
            regime_fallback=bool(cc.get("regime_fallback_for_direction", False)),
            regime_min_confidence=float(cc.get("regime_min_confidence_for_direction", 0.55)),
        )

    def process(
        self,
        struct_features: np.ndarray,
        knowledge_probs: np.ndarray,
        causal_pred: float,
        close: float,
        atr: float,
        volume: float = 0.0,
        bar_timestamp: str = "",
        consecutive_losses: int = 0,
        time_of_day: tuple[float, float] | None = None,
        tick_bid: float = 0.0,
        tick_ask: float = 0.0,
        position_ctx: dict[str, Any] | None = None,
        lock_forecast_direction: str | None = None,
        macro_features: np.ndarray | list | None = None,
    ) -> ThoughtTrace:
        """
        运行完整认知链路。

        返回 ThoughtTrace，包含校准后的概率、置信度、思维链等。
        """
        if not self.enabled:
            return self._passthrough(knowledge_probs, bar_timestamp)

        self._bar_count += 1
        ts = bar_timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # --- Step 1: 更新语境窗口 ---
        sf = np.asarray(struct_features, dtype=np.float32).reshape(-1)
        if sf.size < 30:
            sf = np.pad(sf, (0, 30 - min(sf.size, 30)))[:30]
        kp = np.asarray(knowledge_probs, dtype=np.float32).reshape(-1)
        if kp.size < 3:
            kp = np.array([0.34, 0.33, 0.33], dtype=np.float32)

        ctx = BarContext(
            timestamp=ts,
            struct_features=sf[:30].copy(),
            knowledge_probs=kp[:3].copy(),
            causal_pred=causal_pred,
            close=close,
            atr=atr,
            volume=volume,
        )

        # --- Step 2: 市场状态识别（当前 bar 先入窗再 detect，避免 len<6 → unknown）---
        self.context.push(ctx)
        regime_name, regime_conf, regime_metrics = self.regime.detect(self.context, sf)
        ctx.regime = regime_name
        if self.context.bars:
            self.context.bars[-1].regime = regime_name

        # --- Step 3: 因果叙事 ---
        # 从结构特征估算因果变量
        macro_shock = self._estimate_macro_shock(sf, ctx, macro_features=macro_features)
        risk_av = self._estimate_risk_aversion(sf)
        dollar = self._estimate_dollar(sf)
        demand = self._estimate_demand(sf)
        price_change = causal_pred

        causal_state, events = self.narrative.update(
            macro_shock, risk_av, dollar, demand, price_change, ts,
        )
        narrative_text = self.narrative.narrative_summary()

        # --- Step 4: 信号交叉验证 ---
        agreement_score, conflicts, signals = self.validator.validate(
            kp[:3], causal_pred, self.regime, sf,
        )

        # --- Step 5: 信心校准 ---
        calibrated_probs, confidence, reasoning = self.calibrator.calibrate(
            kp[:3], agreement_score, self.regime, conflicts, causal_pred,
            lock_direction=lock_forecast_direction,
        )

        # --- Step 6: 风险评估 ---
        closes = self.context.close_sequence(12)
        if len(closes) >= 3:
            returns = np.diff(closes) / np.clip(closes[:-1], 1e-9, None)
            raw_vol = float(np.std(returns))
            if np.isfinite(raw_vol) and raw_vol > 0:
                vol = raw_vol
            elif atr > 0 and close > 0:
                vol = atr / close
            else:
                vol = 0.0005
        else:
            if atr > 0 and close > 0:
                vol = atr / close
            else:
                vol = 0.0005
        calibrated_list = [float(x) for x in calibrated_probs.reshape(-1)[:3]]
        trade_bias = self._infer_trade_bias(calibrated_list, regime_name)

        atr_hist = [float(b.atr) for b in self.context.bars if getattr(b, "atr", 0.0) > 0]
        atr_mean = float(np.mean(atr_hist[-12:])) if len(atr_hist) >= 6 else 0.0

        risk_assessment = self.risk.assess(
            self.regime, confidence, vol, consecutive_losses, time_of_day,
            struct_features=sf,
            close=close,
            atr=atr,
            trade_bias=trade_bias,
            atr_mean=atr_mean,
        )

        # --- Step 7: 反事实分析 ---
        intervention_analysis = self.narrative.intervention_analysis()

        # --- Step 8: 构建思维轨迹 ---
        thought = ThoughtTrace(
            timestamp=ts,
            symbol=self.symbol,
            knowledge_probs=[float(x) for x in kp[:3]],
            causal_pred=round(causal_pred, 6),
            regime=regime_name,
            regime_confidence=round(regime_conf, 4),
            narrative=narrative_text,
            narrative_events=events,
            agreement_score=round(agreement_score, 4),
            conflicts=conflicts,
            intervention_analysis=intervention_analysis,
            calibrated_probs=calibrated_list,
            confidence=round(confidence, 4),
            reasoning_chain=reasoning,
            risk_score=risk_assessment.risk_score,
            risk_warnings=risk_assessment.warnings,
            position_mult=risk_assessment.position_mult,
            should_trade=risk_assessment.should_trade,
            sl_mult=risk_assessment.sl_mult,
            tp_mult=risk_assessment.tp_mult,
            trade_bias=trade_bias,
            ai_sl_price=risk_assessment.sl_price,
            ai_tp_price=risk_assessment.tp_price,
        )

        # ===== Step 9 — tick 入场评估（避免追价）=====
        if trade_bias in ("long", "short") and tick_bid > 0 and tick_ask > 0:
            entry_eval = self._evaluate_entry(
                direction="buy" if trade_bias == "long" else "sell",
                tick_bid=tick_bid,
                tick_ask=tick_ask,
                bar_close=close,
                atr=atr,
                regime=regime_name,
                ai_sl=thought.ai_sl_price,
            )
            thought.ai_entry_price = entry_eval["entry_price"]
            thought.entry_should_wait = entry_eval["should_wait"]
            thought.entry_wait_reason = entry_eval["reason"]
        # ===== 结束 =====

        # --- 日志 ---
        self._log_thought(thought)

        return thought

    def _estimate_macro_shock(
        self,
        struct: np.ndarray,
        ctx: BarContext,
        *,
        macro_features: np.ndarray | list | None = None,
    ) -> float:
        """从 C# 宏观向量或结构特征估算宏观冲击强度。"""
        if macro_features is not None:
            f = np.asarray(macro_features, dtype=np.float64).reshape(-1)
            if f.size >= 8:
                shock = float(f[3]) * float(f[1]) * 2.0 - 1.0
                shock += (float(f[5]) - 0.5) * 0.5
                shock += (float(f[7]) - 0.5) * 0.3
                return float(np.clip(shock, -2.0, 2.0))
        if struct.size > 5:
            trend = float(struct[0])
            vol_proxy = float(struct[min(5, struct.size - 1)])
            return float(np.clip(0.5 * trend + 0.5 * vol_proxy, -2.0, 2.0))
        return 0.0

    def _estimate_risk_aversion(self, struct: np.ndarray) -> float:
        if struct.size > 10:
            return float(np.clip(-struct[10] * 0.3, -1.0, 1.0))
        return 0.0

    def _estimate_dollar(self, struct: np.ndarray) -> float:
        if struct.size > 15:
            return float(np.clip(struct[15] * 0.2, -0.5, 0.5))
        return 0.0

    def _estimate_demand(self, struct: np.ndarray) -> float:
        if struct.size > 20:
            return float(np.clip((struct[20] + struct[0]) * 0.15, -1.0, 1.0))
        return 0.0

    @staticmethod
    def _infer_trade_bias(calibrated_probs: list[float], regime: str) -> str:
        """由校准概率 + 市场状态推断 SL/TP 计算方向（多/空）。"""
        if len(calibrated_probs) >= 3:
            short_p, flat_p, long_p = calibrated_probs[0], calibrated_probs[1], calibrated_probs[2]
            if short_p > long_p and short_p >= flat_p:
                return "short"
            if long_p > short_p and long_p >= flat_p:
                return "long"
        regime = (regime or "").lower()
        if regime == "trending_down":
            return "short"
        if regime == "trending_up":
            return "long"
        return "short" if short_p >= long_p else "long"

    def _passthrough(self, probs: np.ndarray, ts: str = "") -> ThoughtTrace:
        """认知引擎禁用时的直通模式。"""
        p = np.asarray(probs, dtype=np.float32).reshape(-1)[:3]
        return ThoughtTrace(
            timestamp=ts or "",
            symbol=self.symbol,
            knowledge_probs=[float(x) for x in p],
            calibrated_probs=[float(x) for x in p],
            confidence=float(max(p)),
            reasoning_chain="认知引擎未启用（直通模式）",
            should_trade=True,
        )

    def _log_thought(self, thought: ThoughtTrace) -> None:
        """追加思维轨迹到日志文件。"""
        try:
            line = json.dumps(thought.to_dict(), ensure_ascii=False) + "\n"
            with open(self._thought_log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.warning("写入思维轨迹日志失败 path=%s", self._thought_log_path, exc_info=True)

    # ===== P2-2: 平仓评估 — AI 判断是否应该平仓 =====
    def _evaluate_entry(
        self,
        direction: str,
        tick_bid: float,
        tick_ask: float,
        bar_close: float,
        atr: float,
        regime: str,
        ai_sl: float = 0.0,
    ) -> dict[str, Any]:
        """基于 tick 评估理想入场价；追价过高则建议等待。"""
        result: dict[str, Any] = {
            "entry_price": bar_close,
            "should_wait": False,
            "reason": "",
        }
        if tick_bid <= 0 or tick_ask <= 0 or atr <= 0 or bar_close <= 0:
            return result

        mid = (tick_bid + tick_ask) / 2.0
        chase_limit = atr * 0.35
        if regime in ("choppy", "ranging"):
            chase_limit = atr * 0.25

        if direction == "buy":
            ideal = min(bar_close, mid, (tick_bid + bar_close) / 2.0)
            chase = tick_ask - ideal
            if chase > chase_limit:
                result["should_wait"] = True
                result["reason"] = f"Ask={tick_ask:.2f} 高于理想{ideal:.2f} chase={chase:.2f}"
                return result
            result["entry_price"] = round(min(ideal, tick_ask), 5)
        else:
            ideal = max(bar_close, mid, (tick_ask + bar_close) / 2.0)
            chase = ideal - tick_bid
            if chase > chase_limit:
                result["should_wait"] = True
                result["reason"] = f"Bid={tick_bid:.2f} 低于理想{ideal:.2f} chase={chase:.2f}"
                return result
            result["entry_price"] = round(max(ideal, tick_bid), 5)

        if ai_sl > 0:
            if direction == "buy" and result["entry_price"] <= ai_sl:
                result["should_wait"] = True
                result["reason"] = "入场价低于智能体止损"
            elif direction == "sell" and result["entry_price"] >= ai_sl:
                result["should_wait"] = True
                result["reason"] = "入场价高于智能体止损"
        return result

    def _regime_metrics(self, struct_features: np.ndarray) -> dict[str, float]:
        """RegimeDetector.detect 返回 (name, conf, metrics) 三元组。"""
        sf = np.asarray(struct_features, dtype=np.float32).reshape(-1)
        _, _, metrics = self.regime.detect(self.context, sf)
        return metrics if isinstance(metrics, dict) else {}

    def sl_tp_for_direction(
        self,
        direction: str,
        thought: ThoughtTrace,
        struct_features: np.ndarray,
        close: float,
        atr: float,
        entry_anchored: bool = False,
    ) -> tuple[float, float]:
        """按实际交易方向返回 SL/TP；trade_bias 不一致时重算。"""
        bias = "long" if direction == "buy" else "short"
        if (
            not entry_anchored
            and bias == thought.trade_bias
            and thought.ai_sl_price > 0
            and thought.ai_tp_price > 0
        ):
            return thought.ai_sl_price, thought.ai_tp_price
        sf = np.asarray(struct_features, dtype=np.float32).reshape(-1)
        ra = self.risk.assess(
            self.regime,
            thought.confidence,
            0.0005,
            0,
            None,
            struct_features=sf,
            close=close,
            atr=atr,
            trade_bias=bias,
            entry_anchored=entry_anchored,
            atr_mean=0.0,
        )
        return ra.sl_price, ra.tp_price

    def sl_price_for_direction(
        self,
        direction: str,
        thought: ThoughtTrace,
        struct_features: np.ndarray,
        close: float,
        atr: float,
    ) -> float:
        """按实际交易方向返回 SL；trade_bias 与 RL 不一致时重算。"""
        sl, _ = self.sl_tp_for_direction(direction, thought, struct_features, close, atr)
        return sl

    def evaluate_position_management(
        self,
        thought: ThoughtTrace,
        position_ctx: dict[str, Any],
        struct_features: np.ndarray,
        close: float,
        atr: float,
        rl_action: int = 0,
        tick_bid: float = 0.0,
        tick_ask: float = 0.0,
        horizon_direction: str = "",
        horizon_confidence: float = 0.0,
        kn2_dec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """M5 持仓管理：结构 SL/TP、trail_mode、扩展 exit_assessment（波段/Horizon/KN2）。"""
        pos_dir = str(position_ctx.get("direction") or "")
        if pos_dir not in ("buy", "sell"):
            return {
                "exit_score": 0.0,
                "exit_reason": "",
                "reasoning": "",
                "trail_mode": "hold",
                "suggested_trailing_sl": 0.0,
                "ai_sl_price": 0.0,
                "ai_tp_price": 0.0,
                "position_mgmt_reason": "",
            }

        sf = np.asarray(struct_features, dtype=np.float32).reshape(-1)
        regime_metrics = self._regime_metrics(sf)
        pos_in_range = float(regime_metrics.get("pos_in_range", 0.5))

        exit_eval = self._evaluate_exit(
            context=self.context,
            calibrated_probs=thought.calibrated_probs,
            confidence=thought.confidence,
            regime=thought.regime,
            close=close,
            atr=atr,
            position_ctx=position_ctx,
            tick_bid=tick_bid,
            tick_ask=tick_ask,
            rl_action=rl_action,
            horizon_direction=horizon_direction,
            horizon_confidence=horizon_confidence,
            kn2_dec=kn2_dec,
            pos_in_range=pos_in_range,
            struct_features=sf,
        )

        trade_dir = "buy" if pos_dir == "buy" else "sell"
        struct_sl, struct_tp = self.sl_tp_for_direction(
            trade_dir, thought, sf, close, atr, entry_anchored=False
        )

        entry = float(position_ctx.get("entry") or 0.0)
        current_sl = float(position_ctx.get("sl") or 0.0)
        current_tp = float(position_ctx.get("tp") or 0.0)
        profit_pct = float(position_ctx.get("profit_pct") or 0.0)
        peak_profit = float(position_ctx.get("peak_profit_pct") or profit_pct)
        hold_seconds = float(position_ctx.get("hold_seconds") or 0.0)
        min_hold = float(position_ctx.get("min_hold_seconds_before_trailing") or 60.0)

        trail_mode = "hold"
        mgmt_reasons: list[str] = []
        suggested_sl = 0.0
        ai_sl = 0.0
        ai_tp = struct_tp if struct_tp > 0 else current_tp

        trend_with_pos = (
            (pos_dir == "buy" and thought.regime in ("trending_up", "breakout_up"))
            or (pos_dir == "sell" and thought.regime in ("trending_down", "breakout_down"))
        )
        wave_extreme = (pos_dir == "buy" and pos_in_range >= 0.88) or (
            pos_dir == "sell" and pos_in_range <= 0.12
        )
        h_dir = str(horizon_direction or "").lower()
        horizon_against = (pos_dir == "buy" and h_dir in ("short", "flat")) or (
            pos_dir == "sell" and h_dir in ("long", "flat")
        )

        min_profit_for_trail = 0.15
        if hold_seconds < min_hold:
            trail_mode = "hold"
            mgmt_reasons.append(f"持仓{hold_seconds:.0f}s<{min_hold:.0f}s暂不动止损")
        elif profit_pct < min_profit_for_trail:
            trail_mode = "hold"
            mgmt_reasons.append(f"浮盈{profit_pct:.2f}%<{min_profit_for_trail}%观察")
        elif wave_extreme and (horizon_against or thought.regime in ("choppy", "ranging")):
            trail_mode = "tighten"
            mgmt_reasons.append(
                f"波段末端 pos={pos_in_range:.2f} regime={thought.regime}"
            )
        elif trend_with_pos and not wave_extreme and profit_pct >= 0.25:
            trail_mode = "run"
            mgmt_reasons.append(f"顺势波段运行 pos={pos_in_range:.2f}")
        elif profit_pct >= 0.6 and peak_profit >= 0.8:
            trail_mode = "tighten"
            mgmt_reasons.append(f"浮盈回撤保护 peak={peak_profit:.2f}%")
        elif profit_pct >= 0.25:
            trail_mode = "run"
            mgmt_reasons.append("结构允许正向移损")
        else:
            trail_mode = "hold"
            mgmt_reasons.append("波动不足保持原止损")

        if trail_mode in ("run", "tighten") and struct_sl > 0:
            if pos_dir == "buy":
                base = current_sl if current_sl > 0 else struct_sl
                candidate = struct_sl if struct_sl > base else base
                if entry > 0 and candidate < entry and profit_pct < 0.5:
                    candidate = base
                suggested_sl = max(base, candidate)
                ai_sl = suggested_sl
            else:
                base = current_sl if current_sl > 0 else struct_sl
                candidate = struct_sl if struct_sl < base else base
                if entry > 0 and candidate > entry and profit_pct < 0.5:
                    candidate = base
                suggested_sl = min(base, candidate) if base > 0 else candidate
                ai_sl = suggested_sl

        if trail_mode == "run" and struct_tp > 0:
            if pos_dir == "buy":
                ai_tp = max(current_tp, struct_tp) if current_tp > 0 else struct_tp
            else:
                ai_tp = min(current_tp, struct_tp) if current_tp > 0 else struct_tp

        if kn2_dec and float(kn2_dec.get("confidence", 0)) >= 0.55:
            kn2_action = str(kn2_dec.get("action_name") or kn2_dec.get("action") or "").lower()
            if kn2_action in ("close", "flat", "hold") and wave_extreme:
                mgmt_reasons.append(f"KN2建议{kn2_action}")

        position_mgmt_reason = "; ".join(mgmt_reasons[:3])
        return {
            **exit_eval,
            "trail_mode": trail_mode,
            "suggested_trailing_sl": round(suggested_sl, 5) if suggested_sl > 0 else 0.0,
            "ai_sl_price": round(ai_sl, 5) if ai_sl > 0 else 0.0,
            "ai_tp_price": round(ai_tp, 5) if ai_tp > 0 else 0.0,
            "position_mgmt_reason": position_mgmt_reason,
            "pos_in_range": round(pos_in_range, 4),
        }

    def evaluate_exit_for_position(
        self,
        thought: ThoughtTrace,
        position_ctx: dict[str, Any],
        rl_action: int,
        close: float,
        atr: float,
        tick_bid: float = 0.0,
        tick_ask: float = 0.0,
        horizon_direction: str = "",
        horizon_confidence: float = 0.0,
        kn2_dec: dict[str, Any] | None = None,
        struct_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """持仓出场评估：须在 RL 决策之后调用，传入 rl_action。"""
        sf = np.asarray(struct_features, dtype=np.float32).reshape(-1) if struct_features is not None else None
        pos_in_range = 0.5
        if sf is not None:
            pos_in_range = float(self._regime_metrics(sf).get("pos_in_range", 0.5))
        return self._evaluate_exit(
            context=self.context,
            calibrated_probs=thought.calibrated_probs,
            confidence=thought.confidence,
            regime=thought.regime,
            close=close,
            atr=atr,
            position_ctx=position_ctx,
            tick_bid=tick_bid,
            tick_ask=tick_ask,
            rl_action=rl_action,
            horizon_direction=horizon_direction,
            horizon_confidence=horizon_confidence,
            kn2_dec=kn2_dec,
            pos_in_range=pos_in_range,
            struct_features=sf,
        )

    def _evaluate_exit(
        self,
        context: ContextWindow,
        calibrated_probs: list[float],
        confidence: float,
        regime: str,
        close: float,
        atr: float,
        position_ctx: dict[str, Any] | None = None,
        tick_bid: float = 0.0,
        tick_ask: float = 0.0,
        rl_action: int = 0,
        horizon_direction: str = "",
        horizon_confidence: float = 0.0,
        kn2_dec: dict[str, Any] | None = None,
        pos_in_range: float = 0.5,
        struct_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        评估是否应该平仓。
        rl_action: 0=hold 1=long 2=short 5=close（与 ACTION_NAMES 一致）
        """
        result: dict[str, Any] = {
            "exit_score": 0.0,
            "exit_reason": "",
            "reasoning": "",
        }

        if close <= 0 or atr <= 0:
            return result

        reasons: list[str] = []
        score = 0.0

        pos_dir = ""
        profit_pct = 0.0
        hold_minutes = 0.0
        if position_ctx:
            pos_dir = str(position_ctx.get("direction") or "")
            profit_pct = float(position_ctx.get("profit_pct") or 0.0)
            hold_minutes = float(position_ctx.get("hold_seconds") or 0.0) / 60.0
            max_hold_min = float(position_ctx.get("max_hold_minutes") or 240.0)
            time_expired = bool(position_ctx.get("time_expired"))
            entry = float(position_ctx.get("entry") or 0.0)
            sl = float(position_ctx.get("sl") or 0.0)

            # 时间止损：仅无浮盈时强制；有浮盈且顺势则延长，不因此出场
            hold_due = time_expired or (max_hold_min > 0 and hold_minutes >= max_hold_min)
            if hold_due:
                trend_with_pos = (
                    (pos_dir == "buy" and regime in ("trending_up", "breakout_up"))
                    or (pos_dir == "sell" and regime in ("trending_down", "breakout_down"))
                )
                if profit_pct <= 0:
                    score += 0.85
                    reasons.append(f"持仓{hold_minutes:.0f}min未盈利时间止损")
                elif trend_with_pos and profit_pct > 0:
                    reasons.append(f"到期浮盈{profit_pct:.2f}%顺势延长")
                elif profit_pct > 0:
                    score += 0.2
                    reasons.append(f"到期浮盈{profit_pct:.2f}%观察出场")

            # 逆势位移用 bid（卖单用 ask 会把点差误判为升破入场）
            mark = tick_bid if tick_bid > 0 else (tick_ask if tick_ask > 0 else close)

            if pos_dir == "buy" and len(calibrated_probs) >= 3:
                short_p, flat_p, long_p = calibrated_probs[0], calibrated_probs[1], calibrated_probs[2]
                rl_still_long = rl_action == 1
                if rl_action == 5:
                    score += 0.45
                    reasons.append("RL建议平仓")
                elif rl_action == 2:
                    score += 0.35
                    reasons.append("RL翻空")
                elif not rl_still_long and short_p > long_p + 0.15 and short_p > 0.50:
                    score += 0.35
                    reasons.append("持仓多但KN偏空/观望")
                if sl > 0 and entry > 0 and mark > 0 and mark < entry:
                    risk_dist = entry - sl
                    adverse = entry - mark
                    if risk_dist > 0 and adverse >= risk_dist * 0.35:
                        score += 0.25
                        reasons.append("价格向止损位移")
            elif pos_dir == "sell" and len(calibrated_probs) >= 3:
                short_p, flat_p, long_p = calibrated_probs[0], calibrated_probs[1], calibrated_probs[2]
                rl_still_short = rl_action in (2, 3, 4)
                if rl_action == 5:
                    score += 0.45
                    reasons.append("RL建议平仓")
                elif rl_action == 1:
                    score += 0.35
                    reasons.append("RL翻多")
                elif not rl_still_short and long_p > short_p + 0.15 and long_p > 0.50:
                    score += 0.35
                    reasons.append("持仓空但KN偏多/观望")
                if sl > 0 and entry > 0 and mark > 0 and mark > entry:
                    risk_dist = sl - entry
                    adverse = mark - entry
                    if risk_dist > 0 and adverse >= risk_dist * 0.35:
                        score += 0.25
                        reasons.append("价格向止损位移")

            if profit_pct >= 0.5 and regime in ("choppy", "ranging", "unknown"):
                score += 0.2
                reasons.append(f"震荡市浮盈{profit_pct:.2f}%保护")
            if profit_pct >= 1.0 and hold_minutes >= 30:
                closes = context.close_sequence(8)
                if len(closes) >= 8:
                    recent = closes[-1] - closes[-4]
                    against = (pos_dir == "buy" and recent < -atr * 0.2) or (pos_dir == "sell" and recent > atr * 0.2)
                    if against:
                        score += 0.25
                        reasons.append("浮盈后反向波动")

        # 方向/观望因子：须与 RL 方向一致才计入（避免 KN 与 RL 冲突误平）
        if len(calibrated_probs) >= 3:
            short_prob = calibrated_probs[0]
            flat_prob = calibrated_probs[1]
            long_prob = calibrated_probs[2]

            if flat_prob > 0.5 and confidence > 0.5 and rl_action in (0, 5):
                if not position_ctx or profit_pct >= 0.8 or hold_minutes >= 15:
                    score += 0.6
                    reasons.append(f"观望概率={flat_prob:.0%} > 50%")
                elif position_ctx and profit_pct < 0.5:
                    reasons.append(f"观望概率高但持仓初期({profit_pct:.2f}%)暂不出场")

            max_dir = max(long_prob, short_prob)
            if max_dir > 0.3 and abs(long_prob - short_prob) < 0.05 and rl_action == 0:
                score += 0.3
                reasons.append("多空方向犹豫")

        # 震荡状态：仅无持仓或已有浮盈保护时加分（持仓初期不因 choppy 单独出场）
        if regime in ("choppy", "unknown"):
            if not position_ctx or profit_pct >= 0.5:
                score += 0.2
                reasons.append(f"市场状态={regime}")
            elif position_ctx and profit_pct < 0.3 and hold_minutes < 10:
                reasons.append(f"震荡初期({hold_minutes:.0f}min)不因choppy单独出场")

        closes = context.close_sequence(6)
        if len(closes) >= 6:
            recent_volatility = float(np.std(closes))
            if recent_volatility > 0:
                recent_trend = closes[-1] - closes[-6]
                if abs(recent_trend) < recent_volatility * 0.3 and not position_ctx:
                    score += 0.1
                    reasons.append("近期窄幅震荡")

        atr_hist = [float(b.atr) for b in context.bars if getattr(b, "atr", 0.0) > 0]
        atr_regime_high = False
        if len(atr_hist) >= 6:
            atr_mean = float(np.mean(atr_hist[-12:]))
            if atr_mean > 0 and atr > atr_mean * 1.8:
                atr_regime_high = True
                score += 0.2
                reasons.append("ATR异常扩大")

        # ===== 波段/Horizon/KN2 扩展出场 =====
        if position_ctx and pos_dir in ("buy", "sell"):
            h_dir = str(horizon_direction or "").lower()
            h_conf = float(horizon_confidence or 0.0)
            wave_top = (pos_dir == "buy" and pos_in_range >= 0.90) or (
                pos_dir == "sell" and pos_in_range <= 0.10
            )
            if wave_top and profit_pct >= 0.3:
                score += 0.25
                reasons.append(f"波段末端 pos={pos_in_range:.2f}")

            if h_conf >= 0.48:
                if pos_dir == "buy" and h_dir in ("flat", "short"):
                    score += 0.2 if h_dir == "flat" else 0.35
                    reasons.append(f"Horizon={h_dir}({h_conf:.0%})")
                elif pos_dir == "sell" and h_dir in ("flat", "long"):
                    score += 0.2 if h_dir == "flat" else 0.35
                    reasons.append(f"Horizon={h_dir}({h_conf:.0%})")

            if kn2_dec:
                kn2_conf = float(kn2_dec.get("confidence", 0))
                kn2_action = str(kn2_dec.get("action_name") or kn2_dec.get("action") or "").lower()
                if kn2_conf >= 0.55:
                    if kn2_action in ("close", "flat") or (
                        pos_dir == "buy" and kn2_action in ("short", "sell")
                    ) or (pos_dir == "sell" and kn2_action in ("long", "buy")):
                        score += 0.3
                        reasons.append(f"KN2={kn2_action}({kn2_conf:.0%})")

            if wave_top and profit_pct >= 0.5 and (
                h_dir in ("flat", "") or regime in ("ranging", "choppy")
            ):
                score += 0.2
                reasons.append("波段走完+观望/震荡")

            if struct_features is not None and struct_features.size >= 4 and atr > 0:
                if pos_dir == "buy":
                    res = self.risk._extract_resistance(struct_features, close, atr)
                    if res > 0 and close >= res - atr * 0.15 and profit_pct >= 0.4:
                        score += 0.15
                        reasons.append(f"接近结构阻力{res:.2f}")
                else:
                    sup = self.risk._extract_support(struct_features, close, atr)
                    if sup > 0 and close <= sup + atr * 0.15 and profit_pct >= 0.4:
                        score += 0.15
                        reasons.append(f"接近结构支撑{sup:.2f}")

            trend_with_pos = (
                (pos_dir == "buy" and regime in ("trending_up", "breakout_up"))
                or (pos_dir == "sell" and regime in ("trending_down", "breakout_down"))
            )
            if atr_regime_high and profit_pct >= 0.4 and not trend_with_pos:
                score += 0.15
                reasons.append("高波动非顺势保护")

        result["exit_score"] = round(min(score, 1.0), 4)
        result["exit_reason"] = "; ".join(reasons[:3]) if reasons else ""
        result["reasoning"] = " | ".join(reasons) if reasons else "继续持有"
        return result
    # ===== 结束 =====

    def record_outcome(self, was_correct: bool) -> None:
        """记录一笔交易的正确性，用于信心校准。"""
        self.calibrator.record_outcome(was_correct)

    def get_context_summary(self) -> dict[str, Any]:
        """获取当前语境摘要（供外部诊断）。"""
        return {
            "bar_count": self._bar_count,
            "context_bars": len(self.context),
            "regime": self.regime.current,
            "regime_confidence": self.regime.confidence,
            "narrative": self.narrative.narrative_summary(),
            "causal_history": self.narrative.history_dicts()[-5:],
            "thought_log": str(self._thought_log_path),
        }

    def rebuild_context_from_m5(self, m5: "pd.DataFrame") -> None:
        """
        从 M5 历史重建语境窗口。
        子进程每 tick 重建，避免 ContextWindow 为空导致 regime=unknown。
        """
        import pandas as pd

        from zhulong.strategies.indicators import atr_series

        if m5 is None or len(m5) < 2:
            return

        cap = self.context.max_bars
        atr_s = atr_series(m5)
        hist = m5.iloc[-(cap + 1) : -1] if len(m5) > cap + 1 else m5.iloc[:-1]
        self.context.bars.clear()

        closes = m5["close"].astype(float).values
        for ts, row in hist.iterrows():
            bar_key = str(ts)
            loc = m5.index.get_loc(ts)
            if isinstance(loc, np.ndarray):
                idx = int(loc[-1])
            elif isinstance(loc, slice):
                idx = int(loc.start) if loc.start is not None else 0
            else:
                idx = int(loc)
            close = float(row["close"])
            atr_val = float(atr_s.iloc[idx]) if idx < len(atr_s) and not pd.isna(atr_s.iloc[idx]) else close * 0.001
            start = max(0, idx - 6)
            trend = (closes[idx] - closes[start]) / max(closes[start], 1e-9)
            sf = np.zeros(30, dtype=np.float32)
            sf[0] = float(np.clip(trend * 50.0, -1.0, 1.0))
            vol = float(row["volume"]) if "volume" in row.index else 0.0
            self.context.bars.append(
                BarContext(
                    timestamp=bar_key,
                    struct_features=sf,
                    knowledge_probs=np.array([0.34, 0.33, 0.33], dtype=np.float32),
                    causal_pred=0.0,
                    close=close,
                    atr=atr_val,
                    volume=vol,
                )
            )

    def resolve_sticky_direction(
        self,
        calibrated_probs: list[float],
        bar_timestamp: str,
    ) -> tuple[str, str, str]:
        """跨 tick 维持的主观点：(sticky_dir, smoothed_regime, instant_dir)。"""
        return self._direction_view.update(
            bar_timestamp,
            calibrated_probs,
            self.regime.current,
            float(self.regime.confidence),
        )

    def export_state(self) -> dict[str, Any]:
        ctx = self.context.to_dict_list()
        last_ctx_ts = ctx[-1].get("timestamp") if ctx else ""
        return {
            "context": ctx,
            "bar_count": self._bar_count,
            "regime": self.regime.current,
            "regime_confidence": self.regime.confidence,
            "last_context_timestamp": last_ctx_ts,
            "thought_log": str(self._thought_log_path),
            "direction_view": self._direction_view.to_dict(),
        }

    def import_state(self, blob: dict[str, Any] | None) -> None:
        if not blob:
            return
        self._bar_count = int(blob.get("bar_count") or 0)
        # 语境窗口每 tick 从 M5 重建；不从磁盘恢复，避免多机 agent_state 漂移
        if blob.get("regime"):
            self.regime._current = str(blob["regime"])
            self.regime._confidence = float(blob.get("regime_confidence") or 0.0)
        self._direction_view.load(blob.get("direction_view"))
        if self._direction_view.smoothed_regime not in ("", "unknown"):
            self.regime._current = self._direction_view.smoothed_regime
