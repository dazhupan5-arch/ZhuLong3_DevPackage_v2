"""因果推理与反事实预测（Phase 4/5）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
try:
    import yaml
except ImportError:
    yaml = None  # optional dependency

logger = logging.getLogger(__name__)

DEFAULT_GRAPH = Path(__file__).resolve().parent.parent.parent / "config" / "causal_graph.yaml"


def load_causal_graph(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_GRAPH
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8-sig")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # Fallback: try JSON equivalent
    json_path = p.with_suffix(".json")
    if json_path.is_file():
        import json
        return json.loads(json_path.read_text(encoding="utf-8-sig"))
    return {}


class CausalInference:
    """基于拟合系数的 SCM 前向传播，输出价格变化预测（百分比）。"""

    def __init__(
        self,
        coef_path: str | Path = "models/causal_coef.pkl",
        symbol: str = "XAUUSD",
        graph_path: str | Path | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.coef_path = Path(coef_path)
        self.graph = load_causal_graph(graph_path)
        self.coef: dict[str, Any] = {}
        self._ready = False
        if self.coef_path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            import joblib

            blob = joblib.load(self.coef_path)
            if isinstance(blob, dict) and self.symbol in blob:
                self.coef = blob[self.symbol]
            elif isinstance(blob, dict) and "price_change" in blob:
                self.coef = blob
            else:
                self.coef = blob if isinstance(blob, dict) else {}
            self._ready = bool(self.coef)
        except Exception as ex:
            logger.warning("因果系数加载失败 %s: %s", self.coef_path, ex)
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict_price_change(
        self,
        macro_shock: float,
        *,
        risk_aversion: float | None = None,
        dollar_index: float | None = None,
    ) -> float:
        """返回未来收益率预测（百分比，正=看多）。"""
        if not self._ready:
            return float(macro_shock) * 0.01

        c = self.coef
        risk = (
            float(risk_aversion)
            if risk_aversion is not None
            else float(c.get("risk_aversion", {}).get("intercept", 0.0))
            + float(c.get("risk_aversion", {}).get("macro_shock", 0.0)) * macro_shock
        )
        dollar = (
            float(dollar_index)
            if dollar_index is not None
            else float(c.get("dollar_index", {}).get("intercept", 0.0))
            + float(c.get("dollar_index", {}).get("macro_shock", 0.0)) * macro_shock
        )
        demand_key = "gold_demand" if self.symbol == "XAUUSD" else "oil_demand"
        demand_cfg = c.get(demand_key, {})
        demand = float(demand_cfg.get("intercept", 0.0))
        demand += float(demand_cfg.get("risk_aversion", 0.0)) * risk
        demand += float(demand_cfg.get("dollar_index", 0.0)) * dollar

        pc = c.get("price_change", {})
        price_change = float(pc.get("intercept", 0.0))
        pc_parent = demand_key.replace("_demand", "_demand")  # gold_demand / oil_demand
        price_change += float(pc.get(pc_parent, pc.get("gold_demand", pc.get("oil_demand", 0.0)))) * demand
        return float(price_change)

    def macro_shock_from_bar(
        self,
        struct_features: np.ndarray | None = None,
        *,
        event_surprise: float = 0.0,
        macro_features: np.ndarray | list | None = None,
    ) -> float:
        """从结构特征、事件惊喜度或 C# 8 维宏观向量估计宏观冲击。"""
        if macro_features is not None:
            return self.macro_shock_from_features(macro_features)
        if abs(event_surprise) > 1e-9:
            return float(event_surprise)
        if struct_features is not None and struct_features.size > 0:
            vol = float(struct_features.reshape(-1)[min(5, struct_features.size - 1)])
            trend = float(struct_features.reshape(-1)[0])
            return float(np.clip(0.6 * trend + 0.4 * vol, -3.0, 3.0))
        return 0.0

    def macro_shock_from_features(self, features: np.ndarray | list) -> float:
        """C# MacroFeatureBuilder 8 维 → 宏观冲击标量。"""
        f = np.asarray(features, dtype=np.float64).reshape(-1)
        if f.size < 8:
            return 0.0
        shock = float(f[3]) * float(f[1]) * 2.0 - 1.0
        shock += (float(f[5]) - 0.5) * 0.6
        shock += (float(f[7]) - 0.5) * 0.4
        if float(f[0]) < 0.15:
            shock += 0.25
        return float(np.clip(shock, -3.0, 3.0))


class CounterfactualPredictor:
    """持仓期间反事实「若不操作」收益估计。"""

    def __init__(self, causal: CausalInference | None = None) -> None:
        self.causal = causal

    def luck_pnl_r(
        self,
        *,
        direction: float,
        entry_price: float,
        position_frac: float,
        initial_balance: float,
        exogenous_sum: float = 0.0,
        hold_bars: int = 1,
    ) -> float:
        """
        估计持仓期间「外生运气」带来的 PnL（R），用于从实际奖励中扣除。
        exogenous_sum: 持仓期间宏观冲击累积。
        """
        sign = 1.0 if direction > 0 else -1.0
        if self.causal and self.causal.is_ready:
            shock = exogenous_sum / max(hold_bars, 1)
            pred_pct = self.causal.predict_price_change(shock) * hold_bars
            exogenous_move = entry_price * (pred_pct / 100.0)
            luck_pnl = exogenous_move * sign * abs(position_frac)
        else:
            luck_pnl = 0.0
        return float(luck_pnl / max(initial_balance, 1e-9))

    def causal_reward(self, actual_pnl_r: float, luck_pnl_r: float) -> float:
        return float(actual_pnl_r - luck_pnl_r)


def fuse_knowledge_with_causal(
    knowledge_probs: np.ndarray,
    causal_pred: float,
    *,
    weight_knowledge: float = 0.7,
    weight_causal: float = 0.3,
) -> np.ndarray:
    """
    后处理融合：knowledge_probs 顺序 [空/short, 观望/flat, 多/long]（与 KnowledgeNet 训练一致）。
    causal_pred: 百分比收益率预测。
    """
    p = np.asarray(knowledge_probs, dtype=np.float32).reshape(-1)
    if p.size < 3:
        p = np.array([0.34, 0.33, 0.33], dtype=np.float32)
    short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])

    if causal_pred > 0.02:
        boost = min(weight_causal, 0.5)
        long_p += boost
        short_p -= boost * 0.5
    elif causal_pred < -0.02:
        boost = min(weight_causal, 0.5)
        short_p += boost
        long_p -= boost * 0.5

    fused = np.array([short_p, flat_p, long_p], dtype=np.float32)
    fused = np.clip(fused, 1e-6, None)
    fused /= fused.sum()
    wk = float(weight_knowledge)
    wc = float(weight_causal)
    blend = wk * p[:3] + wc * fused
    blend = np.clip(blend, 1e-6, None)
    blend /= blend.sum()
    return blend.reshape(1, -1)


def save_causal_coef(coef_by_symbol: dict[str, Any], path: str | Path) -> None:
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(coef_by_symbol, out)
    meta = {sym: {k: v for k, v in blob.items() if k != "fit_stats"} for sym, blob in coef_by_symbol.items()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
