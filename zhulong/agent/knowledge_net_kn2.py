"""KN 2.0 —— 交易员认知叙事网络（GRU + 多情景预测 + 端到端决策）。

设计理念：
  GRU 阅读当下（过去 N 根 K 线的结构演变） → 产生 h_t（连续认知）
  → 预测模块（基于 h_t 推演多种可能未来情景）
  → 决策模块（在所有情景中找到最优行动方案）
  → 输出：动作 + 仓位 + SL + TP + 信心

KN = 交易员的大脑（全权决策），PPO = 交易员的手（纯执行）。

和 KN 1.0 (ResNet) 的核心区别：
  - ResNet → GRU：隐藏状态跨 K 线传递，思维连续不跳变
  - 三分类 → 多任务端到端：不是预测方向，是预测完整交易决策
  - 静态前向 → 有状态推理：每步传入上一步的认知状态
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_torch = None
_nn = None
_TORCH_ERR: Exception | None = None


def _ensure_torch():
    global _torch, _nn, _TORCH_ERR
    if _torch is not None:
        return _torch, _nn
    if _TORCH_ERR is not None:
        raise _TORCH_ERR
    try:
        import torch
        import torch.nn as nn

        _torch = torch
        _nn = nn
        return torch, nn
    except (ImportError, OSError) as ex:
        _TORCH_ERR = ex
        raise


def _normalize_action_class_weights(
    class_weights: list[float] | None,
    num_actions: int = 6,
) -> list[float] | None:
    """hold/long/short 可传 3 维，自动补齐 num_actions 维。"""
    if not class_weights:
        return None
    w = [float(x) for x in class_weights]
    if len(w) == 3 and num_actions > 3:
        w = w + [1.0] * (num_actions - 3)
    if len(w) != num_actions:
        raise ValueError(f"class_weights len {len(w)} != num_actions {num_actions}")
    return w


# ==============================================================================
# GRU 认知叙事网络
# ==============================================================================


def _build_trader_gru_class(
    hidden_dim: int = 256,
    num_layers: int = 2,
    embed_dim: int = 64,
    num_actions: int = 6,
    market_dim: int = 98,
):
    """构建 TraderKnowledgeGRU 类。"""
    torch, nn = _ensure_torch()

    class TraderKnowledgeGRU(nn.Module):
        """
        交易员认知 GRU 网络。

        输入 (每根 K 线):
          market_features[market_dim] — legacy: V14(68)+struct(30)=98; v16: struct(30)+horizon(35)=65
          position_state[6]   = [direction_norm, hold_bars_log, float_pnl_pct,
                                 max_fav_pct, max_adv_pct, is_holding]

        输出:
          action_logits[num_actions]  — hold/long/short/...
          position_size     — 仓位比例 0~1
          sl_atr_mult       — 止损 ATR 倍数 0.5~3.5
          tp_atr_mult       — 止盈 ATR 倍数 1.0~6.0
          confidence        — 决策信心 0~1
          should_trade      — 是否交易
          scenarios[64]     — 多情景预测（辅助训练）
          embedding[64]     — 认知嵌入（给 PPO）
          hidden            — GRU 隐藏状态（延续到下一根 K 线）
        """

        MARKET_DIM = int(market_dim)
        POS_DIM = 6       # position state
        NUM_ACTIONS = num_actions   # hold/long/short (3) or hold/long/short/s50/s100/close (6)
        NUM_SCENARIOS = 8
        SCENARIO_PARAMS = 8  # per scenario: delta_price, delta_vol, touch_sl, touch_tp, etc.

        def __init__(self):
            super().__init__()
            self.market_dim = self.MARKET_DIM
            self.pos_dim = self.POS_DIM
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.embed_dim = embed_dim

            # 市场编码器
            self.market_encoder = nn.Sequential(
                nn.Linear(self.MARKET_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # 持仓编码器
            self.pos_encoder = nn.Sequential(
                nn.Linear(self.POS_DIM, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # GRU 核心
            gru_input_dim = hidden_dim + hidden_dim // 2
            self.gru = nn.GRU(
                gru_input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=False,
                dropout=0.1 if num_layers > 1 else 0.0,
            )

            # 动作头
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.NUM_ACTIONS),
            )

            # 仓位头
            self.size_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 止损头（ATR 倍数，输出 0.5~3.5）
            self.sl_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 止盈头（ATR 倍数，输出 1.0~6.0）
            self.tp_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 信心头
            self.conf_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 可交易性头
            self.trade_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 多情景预测头（辅助训练）
            scenario_out_dim = self.NUM_SCENARIOS * self.SCENARIO_PARAMS
            self.scenario_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, scenario_out_dim),
            )

            # 认知嵌入头（给 PPO）
            self.embed_head = nn.Linear(hidden_dim, embed_dim)

        def forward(
            self,
            market_feat: "torch.Tensor",
            h_prev: "torch.Tensor | None" = None,
            position_state: "torch.Tensor | None" = None,
        ) -> dict[str, "torch.Tensor"]:
            """
            Args:
                market_feat: (batch, 98) 市场特征
                h_prev: (num_layers, batch, hidden_dim) 上一步 GRU 隐藏状态
                position_state: (batch, 6) 当前持仓状态，无仓时为零向量

            Returns:
                dict with all outputs + 'hidden' for next step
            """
            batch_size = market_feat.shape[0]
            device = market_feat.device

            # 编码市场特征
            m = self.market_encoder(market_feat)  # (batch, hidden_dim)

            # 编码持仓状态
            if position_state is not None:
                p = self.pos_encoder(position_state)  # (batch, hidden_dim//2)
            else:
                p = torch.zeros(batch_size, self.hidden_dim // 2, device=device)

            # 拼接 → GRU
            combined = torch.cat([m, p], dim=-1).unsqueeze(0)  # (1, batch, gru_input)
            output, h_new = self.gru(combined, h_prev)  # output: (1, batch, hidden_dim)
            h_out = output.squeeze(0)  # (batch, hidden_dim)

            return {
                "action_logits": self.action_head(h_out),
                "position_size": torch.sigmoid(self.size_head(h_out)),
                "sl_atr_mult": torch.sigmoid(self.sl_head(h_out)) * 3.0 + 0.5,
                "tp_atr_mult": torch.sigmoid(self.tp_head(h_out)) * 5.0 + 1.0,
                "confidence": torch.sigmoid(self.conf_head(h_out)),
                "should_trade_logit": self.trade_head(h_out),
                "should_trade_prob": torch.sigmoid(self.trade_head(h_out)),
                "scenarios": self.scenario_head(h_out),
                "embedding": self.embed_head(h_out),
                "hidden": h_new,
            }

    return TraderKnowledgeGRU, torch


# ==============================================================================
# Triple Barrier 标签生成
# ==============================================================================


def build_triple_barrier_labels(
    data: np.ndarray,  # shape (n_bars, features) where col 0 = close
    atr: np.ndarray,   # shape (n_bars,)
    *,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    max_hold_bars: int = 48,
    close_col: int = 0,
) -> np.ndarray:
    """
    为每根 K 线生成 Triple Barrier 标签。

    对每根 K 线 t：
      - 上轨 = close[t] + tp_atr_mult * atr[t]
      - 下轨 = close[t] - sl_atr_mult * atr[t]
      - 时间窗 = t + max_hold_bars

    标签：
      2 = 上轨先触及（做多方向成功，波段成立）
      0 = 下轨先触及（做多方向失败，被止损）
      1 = 时间到但未触任何一轨（方向不明确/横盘）

    返回: shape (n_bars,) 的 int 数组
    """
    n = len(data)
    labels = np.full(n, 1, dtype=np.int32)  # 默认=1（时间到）

    for t in range(n):
        upper = data[t, close_col] + tp_atr_mult * atr[t]
        lower = data[t, close_col] - sl_atr_mult * atr[t]
        horizon = min(t + max_hold_bars + 1, n)

        for fwd in range(t + 1, horizon):
            high = data[fwd, close_col]  # simplified: use close as high
            low = data[fwd, close_col]   # simplified: use close as low

            # 更好的做法是用实际 high/low，但这里先简化为 close
            # 实际使用时应该传入 high/low 列
            if high >= upper:
                labels[t] = 2  # TP 先触
                break
            if low <= lower:
                labels[t] = 0  # SL 先触
                break

    return labels


# ==============================================================================
# 持仓状态编码
# ==============================================================================


def encode_position_state(
    direction: float = 0.0,
    hold_bars: int = 0,
    float_pnl_pct: float = 0.0,
    max_favorable_pct: float = 0.0,
    max_adverse_pct: float = 0.0,
    max_hold_bars: int = 48,
) -> np.ndarray:
    """编码持仓状态为 6 维向量。

    Returns:
        np.ndarray shape (6,): [direction_norm, hold_bars_log,
                                float_pnl_pct, max_fav_pct, max_adv_pct, is_holding]
    """
    return np.array(
        [
            direction,                                                 # -1/0/1
            math.log1p(hold_bars) / math.log1p(max_hold_bars),        # 0~1
            np.clip(float_pnl_pct, -1.0, 1.0),
            np.clip(max_favorable_pct, 0.0, 1.0),
            np.clip(max_adverse_pct, 0.0, 1.0),
            1.0 if direction != 0 else 0.0,                           # is_holding
        ],
        dtype=np.float32,
    )


# ==============================================================================
# KN 2.0 推理封装
# ==============================================================================


class KN2Inference:
    """KN 2.0 推理封装 —— 兼容现有 KnowledgeNetInference 模式。

    和 KN 1.0 的区别：
      - predict() 返回决策字典而非 (probs, emb)
      - 维护内部 GRU 隐藏状态
      - 需要传入持仓状态
    """

    def __init__(
        self,
        model_path: str | Path,
        meta_path: str | Path | None = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embed_dim: int = 64,
        dual_channel_cfg: dict | None = None,
        market_dim: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_actions = 6  # default, overridden by meta.json
        self.market_dim = int(market_dim or 98)

        # ── 一致性方向过滤配置 ──
        dc = dual_channel_cfg or {}
        self._dc_enabled = bool(dc.get("enabled", True))
        self._dc_h_norm_max = float(dc.get("h_norm_max", 5.0))
        self._dc_consensus_bars = int(dc.get("consensus_bars", 6))
        self._dc_gap_threshold = float(dc.get("gap_threshold", 0.06))
        self._dc_gap_cooldown_bars = int(dc.get("gap_cooldown_bars", 12))

        meta = {}
        # Auto-discover meta.json beside model file
        if meta_path is None:
            auto_meta = self.model_path.with_suffix(".meta.json")
            if auto_meta.is_file():
                meta = json.loads(auto_meta.read_text(encoding="utf-8"))
                logger.info("KN 2.0 read meta.json: %s", auto_meta)
        elif meta_path:
            mp = Path(meta_path)
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))

        self.hidden_dim = int(meta.get("hidden_dim", hidden_dim))
        self.num_layers = int(meta.get("num_layers", num_layers))
        self.embed_dim = int(meta.get("embed_dim", embed_dim))
        self.num_actions = int(meta.get("num_actions", 6))
        self.market_dim = int(meta.get("market_dim", self.market_dim))

        self.model = None
        self._onnx_session = None
        self._ready = False

        # GRU hidden state (initialized to zero on start)
        self._h: np.ndarray | None = None

        self._load_model()

    def _load_model(self) -> None:
        pth_path = (
            self.model_path
            if self.model_path.suffix.lower() == ".pth"
            else self.model_path.with_suffix(".pth")
        )
        onnx_path = (
            self.model_path
            if self.model_path.suffix.lower() == ".onnx"
            else self.model_path.with_suffix(".onnx")
        )

        # 优先 ONNX
        if onnx_path.is_file():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._onnx_session = ort.InferenceSession(
                    str(onnx_path.resolve()),
                    opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ready = True
                logger.info("KN 2.0 ONNX 已加载: %s", onnx_path)
                return
            except Exception as ex:
                logger.warning("KN 2.0 ONNX 加载失败: %s", ex)

        # PyTorch 回退
        if pth_path.is_file():
            try:
                torch, _ = _ensure_torch()
                KnCls, _ = _build_trader_gru_class(
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_actions=self.num_actions,
                    market_dim=self.market_dim,
                )
                self.model = KnCls()
                state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self._ready = True
                logger.info("KN 2.0 PyTorch 已加载: %s", pth_path)
                return
            except Exception as ex:
                logger.warning("KN 2.0 PyTorch 加载失败: %s", ex)

        logger.warning("KN 2.0 模型未加载，将使用启发式回退")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset_hidden(self) -> None:
        """重置 GRU 隐藏状态（换品种/新会话时调用）。"""
        self._h = None
        # 清除一致性和冷却状态
        self._dc_gap_history = []
        self._dc_gap_ema = None
        self._dc_cooldown = 0

    def predict(
        self,
        market_features: np.ndarray,    # (1, market_dim) or (market_dim,)
        position_state: np.ndarray | None = None,  # (1, 6) or (6,) or None
        close: float | None = None,
        atr: float | None = None,
    ) -> dict[str, Any]:
        """
        单 bar 推理。

        Args:
            market_features: 市场特征向量 (market_dim,)
            position_state: 持仓状态向量 (6,) 或 None（无仓）
            close: 当前收盘价（双通道架构需要真实价格）
            atr: 当前 ATR（双通道架构需要真实波动率）

        Returns:
            dict: {
                "action": int (0-5),
                "action_name": str,
                "position_size": float,
                "sl_atr_mult": float,
                "tp_atr_mult": float,
                "confidence": float,
                "should_trade": bool,
                "embedding": np.ndarray (64,),
                "scenarios": np.ndarray (64,),
            }
        """
        mf = np.asarray(market_features, dtype=np.float32).reshape(1, -1)
        md = int(self.market_dim)
        if mf.shape[1] < md:
            mf = np.pad(mf, ((0, 0), (0, md - mf.shape[1])))
        mf = mf[:, :md]

        if position_state is not None:
            ps = np.asarray(position_state, dtype=np.float32).reshape(1, -1)
            if ps.shape[1] < 6:
                ps = np.pad(ps, ((0, 0), (0, 6 - ps.shape[1])))
            ps = ps[:, :6]
        else:
            ps = np.zeros((1, 6), dtype=np.float32)

        if not self._ready:
            return self._heuristic_predict(mf, ps)

        try:
            if self._onnx_session is not None:
                return self._predict_onnx(mf, ps)
            else:
                return self._predict_torch(mf, ps, close, atr)
        except Exception as ex:
            logger.warning("KN 2.0 推理失败，启发式回退: %s", ex)
            return self._heuristic_predict(mf, ps)

    def _predict_torch(
        self, mf: np.ndarray, ps: np.ndarray,
        close: float | None = None,
        atr: float | None = None,
    ) -> dict[str, Any]:
        torch_mod, _ = _ensure_torch()
        with torch_mod.no_grad():
            mf_t = torch_mod.tensor(mf)
            ps_t = torch_mod.tensor(ps)

            if self._h is not None:
                h_prev = torch_mod.tensor(self._h)
            else:
                h_prev = None

            outputs = self.model(mf_t, h_prev, ps_t)
            h_new = outputs["hidden"]

            if self._dc_enabled:
                try:
                    # ── 一致性方向过滤器：用 action_head 的 softmax 概率差 ——
                    raw = outputs["action_logits"].detach().numpy().ravel().astype(np.float64)
                    probs = np.exp(raw - raw.max()) / np.sum(np.exp(raw - raw.max()))
                    long_p, short_p = float(probs[1]), float(probs[2])
                    gap = long_p - short_p  # + → 偏多, - → 偏空, 0 → 持平
                    action = self._consistency_decide(gap)

                    if action != 0:
                        h_norm = float(torch_mod.norm(h_new).item())
                        if h_norm > self._dc_h_norm_max:
                            h_new = h_new * (self._dc_h_norm_max / h_norm)
                except Exception as ex:
                    logger.warning("KN 2.0 一致性决策失败，回退标准推理: %s", ex)
                    action = int(outputs["action_logits"].argmax(dim=-1)[0].item())
            else:
                action = int(outputs["action_logits"].argmax(dim=-1)[0].item())

            self._h = h_new.numpy()

            return self._format_output(outputs, action)

    # ── 一致性方向过滤（相对基线版） ──
    def _consistency_decide(self, gap: float) -> int:
        """用 action_head 概率差做方向判断，叠加持续性要求。
        gap = prob_long - prob_short，正值偏向多，负值偏向空。
        
        关键改进：不要求绝对值跨越阈值，而是超越自身近期 EMA 基线。
        这样即使模型有结构性 4% 长多偏置，只要 gap 相对自身明显下降，
        就能触发做空信号。
        """
        if not hasattr(self, '_dc_gap_history'):
            self._dc_gap_history = []
        if not hasattr(self, '_dc_gap_ema'):
            self._dc_gap_ema = None
        if not hasattr(self, '_dc_cooldown'):
            self._dc_cooldown = 0

        # EMA 更新
        alpha = 2.0 / (self._dc_consensus_bars + 1)
        if self._dc_gap_ema is None:
            self._dc_gap_ema = gap
        else:
            self._dc_gap_ema = alpha * gap + (1 - alpha) * self._dc_gap_ema

        self._dc_gap_history.append(gap)
        if len(self._dc_gap_history) > self._dc_consensus_bars + 4:
            self._dc_gap_history.pop(0)

        if self._dc_cooldown > 0:
            self._dc_cooldown -= 1
            return 0

        need = self._dc_consensus_bars
        if len(self._dc_gap_history) < need:
            return 0

        # 相对基线的偏差：gap 相对自身 EMA 的偏移
        recent_gaps = self._dc_gap_history[-need:]
        recent_deviations = [g - self._dc_gap_ema for g in recent_gaps]

        all_long_signal = all(d > self._dc_gap_threshold for d in recent_deviations)
        all_short_signal = all(d < -self._dc_gap_threshold for d in recent_deviations)

        if all_long_signal:
            self._dc_cooldown = self._dc_gap_cooldown_bars
            return 1
        if all_short_signal:
            self._dc_cooldown = self._dc_gap_cooldown_bars
            return 2
        return 0

    def _update_return_in_atr(self, close: float, atr: float) -> float:
        """跟踪最近 12 根 bar 的收盘价，返回 (close - close_12) / atr。"""
        if not hasattr(self, '_dc_close_history'):
            self._dc_close_history = []
        self._dc_close_history.append(close)
        if len(self._dc_close_history) > 12:
            self._dc_close_history.pop(0)

        if len(self._dc_close_history) < 12 or atr <= 0:
            return 0.0
        return (self._dc_close_history[-1] - self._dc_close_history[0]) / atr

    def _compute_fact_logits_from_price(self, ret_atr: float) -> np.ndarray:
        """基于真实价格变化的事实动作 logits（非对称版：只能压多或做空，不能做多）。"""
        fact_logits = np.zeros(3, dtype=np.float32)
        fact_logits[0] = abs(ret_atr) * 1.0             # hold: 趋势越强越倾向观望
        fact_logits[1] = 0.0                              # long: 永不为 0——KN2 已有 long 先验
        fact_logits[2] = max(-ret_atr, 0.0) * 2.5        # short: 只有下跌趋势才开
        return fact_logits

    def _compute_gate_from_return(
        self, ret_atr: float, prior_logits: np.ndarray, fact_logits: np.ndarray
    ) -> float:
        """基于价格趋势的门控：趋势越强 + 分歧越大 → gate 越高。
        非对称设计：只对下跌趋势（ret_atr < 0）开门，上涨趋势永远闭门。"""
        # 上涨趋势 → 闭门（KN2 的 long 先验已足够处理上涨）
        if ret_atr >= 0:
            return 0.0

        prior_logits = np.asarray(prior_logits, dtype=np.float64)
        fact_logits = np.asarray(fact_logits, dtype=np.float64)

        # 下跌趋势需要足够强才开门
        trend_clarity = min(abs(ret_atr) * self._dc_gate_sensitivity, 1.0)

        # Jensen-Shannon divergence
        prior_p = np.exp(prior_logits - prior_logits.max()) / np.sum(np.exp(prior_logits - prior_logits.max()))
        fact_p = np.exp(fact_logits - fact_logits.max()) / np.sum(np.exp(fact_logits - fact_logits.max()))
        m_p = (prior_p + fact_p) / 2
        def kl(p, q): return float(np.sum(p * (np.log(p + 1e-9) - np.log(q + 1e-9))))
        jsd = (kl(prior_p, m_p) + kl(fact_p, m_p)) / 2

        raw = (3.0 * trend_clarity + 2.0 * jsd) - self._dc_gate_bias
        gate = float(1.0 / (1.0 + np.exp(-raw)))
        return max(0.0, min(1.0, gate))

    def _predict_onnx(
        self, mf: np.ndarray, ps: np.ndarray
    ) -> dict[str, Any]:
        session = self._onnx_session
        inp_names = [i.name for i in session.get_inputs()]
        out_names = [o.name for o in session.get_outputs()]

        feed = {}
        if "market_feat" in inp_names:
            feed["market_feat"] = mf
        if "position_state" in inp_names:
            feed["position_state"] = ps
        if "h_prev" in inp_names and self._h is not None:
            feed["h_prev"] = self._h.astype(np.float32)

        outputs = session.run(out_names, feed)
        result = dict(zip(out_names, outputs))

        # 更新隐藏状态
        if "hidden" in result:
            self._h = result["hidden"]

        action = int(np.argmax(result.get("action_logits", [[0]*6])[0]))
        return self._format_output_from_onnx(result, action)

    def _format_output(self, outputs: dict, action: int) -> dict[str, Any]:
        ACTION_NAMES = ["hold", "long", "short", "short_50", "short_100", "close"]
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "position_size": float(outputs["position_size"][0, 0]),
            "sl_atr_mult": float(outputs["sl_atr_mult"][0, 0]),
            "tp_atr_mult": float(outputs["tp_atr_mult"][0, 0]),
            "confidence": float(outputs["confidence"][0, 0]),
            "should_trade": bool(outputs["should_trade_prob"][0, 0] > 0.5),
            "embedding": outputs["embedding"][0].numpy(),
            "scenarios": outputs["scenarios"][0].numpy(),
        }

    def _format_output_from_onnx(self, raw: dict, action: int) -> dict[str, Any]:
        ACTION_NAMES = ["hold", "long", "short", "short_50", "short_100", "close"]
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "position_size": float(raw.get("position_size", [[0.0]])[0][0]),
            "sl_atr_mult": float(raw.get("sl_atr_mult", [[1.5]])[0][0]),
            "tp_atr_mult": float(raw.get("tp_atr_mult", [[2.0]])[0][0]),
            "confidence": float(raw.get("confidence", [[0.5]])[0][0]),
            "should_trade": bool(float(raw.get("should_trade_prob", [[0.0]])[0][0]) > 0.5),
            "embedding": np.array(raw.get("embedding", [np.zeros(self.embed_dim)])[0]),
            "scenarios": np.array(raw.get("scenarios", [np.zeros(64)])[0]),
        }

    def _heuristic_predict(self, mf: np.ndarray, ps: np.ndarray) -> dict[str, Any]:
        """启发式回退：基于 V14 特征第一维的趋势信号。"""
        trend = float(mf[0, 0]) if mf.shape[1] > 0 else 0.0

        if trend > 0.3:
            action, conf = 1, min(0.6 + abs(trend) * 0.3, 0.85)
        elif trend < -0.3:
            action, conf = 2, min(0.6 + abs(trend) * 0.3, 0.85)
        else:
            action, conf = 0, 0.4

        return {
            "action": action,
            "action_name": ["hold", "long", "short", "short_50", "short_100", "close"][action],
            "position_size": 1.0,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.0,
            "confidence": conf,
            "should_trade": conf > 0.55,
            "embedding": np.zeros(self.embed_dim, dtype=np.float32),
            "scenarios": np.zeros(64, dtype=np.float32),
        }


# ==============================================================================
# 端到端训练
# ==============================================================================


def train_kn2_end_to_end(
    market_features: np.ndarray,
    position_states: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    val_ratio: float = 0.2,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.0005,
    patience: int = 20,
    hidden_dim: int = 256,
    num_layers: int = 2,
    embed_dim: int = 64,
    num_actions: int = 6,
    market_dim: int | None = None,
    out_path: str | Path = "models/kn2_trader.pth",
    device: str = "auto",
    sequence_length: int = 64,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    """
    KN 2.0 端到端训练。

    训练目标：
      1. 动作分类 — 学习 Triple Barrier 标签（上轨先触=long, 下轨=short, 时间到=flat）
      2. 仓位回归 — 学习最优仓位比例
      3. SL/TP 回归 — 学习最优止损止盈
      4. 信心 — 学习判断自己决策的准确度
      5. 可交易性 — 学习识别高胜率入场点

    训练方式：
      - 将数据切分为固定长度的序列（sequence_length）
      - 每个序列内 GRU 隐藏状态连续传递
      - 序列间重置隐藏状态
      - 多任务联合损失优化
    """
    torch, nn = _ensure_torch()
    from zhulong.utils.device import resolve_torch_device
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    md = int(market_dim or market_features.shape[1])
    device_obj = torch.device(resolve_torch_device(device))
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_actions=num_actions,
        market_dim=md,
    )
    model = KnCls().to(device_obj)

    # 数据准备：切成序列
    n_bars = len(market_features)
    seqs = []
    for start in range(0, n_bars - sequence_length, sequence_length // 2):
        end = min(start + sequence_length, n_bars)
        if end - start < sequence_length // 2:
            continue
        seqs.append((start, end))

    # 划分训练/验证
    n_seqs = len(seqs)
    train_seqs = seqs[: int(n_seqs * (1 - val_ratio))]
    val_seqs = seqs[int(n_seqs * (1 - val_ratio)):]
    print(
        f"  KN2 sequences: {n_seqs:,} (train={len(train_seqs):,} val={len(val_seqs):,}) "
        f"seq_len={sequence_length} device={device_obj.type}",
        flush=True,
    )

    # 损失函数
    cw = _normalize_action_class_weights(class_weights, num_actions)
    action_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device_obj) if cw else None
    )
    reg_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    if cw:
        print(f"  action class_weights={cw}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    stale = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for si, (seq_start, seq_end) in enumerate(train_seqs):
            if ep == 0 and si > 0 and si % 500 == 0:
                print(f"  epoch 1 progress: seq {si:,}/{len(train_seqs):,}", flush=True)
            seq_len = seq_end - seq_start
            mf_seq = torch.tensor(
                market_features[seq_start:seq_end], dtype=torch.float32
            ).to(device_obj)  # (seq_len, 98)

            ps_seq = torch.tensor(
                position_states[seq_start:seq_end], dtype=torch.float32
            ).to(device_obj)  # (seq_len, 6)

            # 逐 bar 前向传播
            h = None
            total_loss = 0.0

            for t in range(seq_len):
                mf_t = mf_seq[t:t+1]  # (1, 98)
                ps_t = ps_seq[t:t+1]  # (1, 6)

                outputs = model(mf_t, h, ps_t)
                h = outputs["hidden"].detach()  # 切断梯度防止 BPTT 过长

                # 多任务损失
                loss = torch.tensor(0.0, device=device_obj)

                # 动作损失
                if "action" in targets and seq_start + t < len(targets["action"]):
                    target_action = torch.tensor(
                        [targets["action"][seq_start + t]], dtype=torch.long
                    ).to(device_obj)
                    loss = loss + action_loss_fn(outputs["action_logits"], target_action)

                # 仓位损失
                if "position_size" in targets and seq_start + t < len(targets["position_size"]):
                    target_size = torch.tensor(
                        [[targets["position_size"][seq_start + t]]], dtype=torch.float32
                    ).to(device_obj)
                    loss = loss + 0.5 * reg_loss_fn(outputs["position_size"], target_size)

                # 止损损失
                if "sl_atr_mult" in targets and seq_start + t < len(targets["sl_atr_mult"]):
                    target_sl = torch.tensor(
                        [[targets["sl_atr_mult"][seq_start + t]]], dtype=torch.float32
                    ).to(device_obj)
                    loss = loss + 0.3 * reg_loss_fn(outputs["sl_atr_mult"], target_sl)

                # 止盈损失
                if "tp_atr_mult" in targets and seq_start + t < len(targets["tp_atr_mult"]):
                    target_tp = torch.tensor(
                        [[targets["tp_atr_mult"][seq_start + t]]], dtype=torch.float32
                    ).to(device_obj)
                    loss = loss + 0.3 * reg_loss_fn(outputs["tp_atr_mult"], target_tp)

                # 可交易性损失
                if "should_trade" in targets and seq_start + t < len(targets["should_trade"]):
                    target_trade = torch.tensor(
                        [[targets["should_trade"][seq_start + t]]], dtype=torch.float32
                    ).to(device_obj)
                    loss = loss + 0.2 * bce_loss_fn(
                        outputs["should_trade_logit"], target_trade
                    )

                total_loss = total_loss + loss

            # 序列平均损失
            seq_loss = total_loss / seq_len
            opt.zero_grad(set_to_none=True)
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += float(seq_loss.item())
            n_train += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_train, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for seq_start, seq_end in val_seqs:
                seq_len = seq_end - seq_start
                mf_seq = torch.tensor(
                    market_features[seq_start:seq_end], dtype=torch.float32
                ).to(device_obj)
                ps_seq = torch.tensor(
                    position_states[seq_start:seq_end], dtype=torch.float32
                ).to(device_obj)

                h = None
                seq_vloss = 0.0

                for t in range(seq_len):
                    mf_t = mf_seq[t:t+1]
                    ps_t = ps_seq[t:t+1]
                    outputs = model(mf_t, h, ps_t)
                    h = outputs["hidden"]

                    loss = torch.tensor(0.0, device=device_obj)
                    if "action" in targets and seq_start + t < len(targets["action"]):
                        ta = torch.tensor([targets["action"][seq_start + t]], dtype=torch.long).to(device_obj)
                        loss = loss + action_loss_fn(outputs["action_logits"], ta)

                    seq_vloss += float(loss.item())

                val_loss += seq_vloss / seq_len
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)

        if ep % 10 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1:3d}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_loss - 0.0001:
            best_loss = avg_val_loss
            stale = 0
            torch.save(model.state_dict(), out)
            meta = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "embed_dim": embed_dim,
                "num_actions": num_actions,
                "market_dim": md,
                "pos_dim": 6,
                "architecture": "kn2_v16" if md == 65 else "kn2_legacy",
                "val_loss": avg_val_loss,
                "class_weights": cw,
            }
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    return {
        "val_loss": best_loss,
        "model_path": str(out),
    }


def train_kn2_fast(
    market_features: np.ndarray,
    position_states: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    val_ratio: float = 0.2,
    epochs: int = 200,
    batch_size: int = 32,   # number of SEQUENCES per batch (not bars)
    lr: float = 0.0005,
    patience: int = 20,
    class_weights: list | None = None,  # [w0, w1, w2] for hold/long/short
    num_actions: int = 6,         # 3 for simple, 6 for full
    hidden_dim: int = 256,
    num_layers: int = 2,
    embed_dim: int = 64,
    out_path: str | Path = "models/kn2_trader.pth",
    device: str = "auto",
    sequence_length: int = 64,
    market_dim: int | None = None,
) -> dict[str, Any]:
    """KN 2.0 fast batched training — multiple sequences processed in parallel.

    Packs N sequences into (seq_len, N, dim) tensors for maximum GRU throughput.
    Same sequence length for all sequences in a batch (padding last if needed).
    """
    torch, nn = _ensure_torch()
    from zhulong.utils.device import resolve_torch_device

    device_obj = torch.device(resolve_torch_device(device))
    md = int(market_dim or market_features.shape[1])
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=hidden_dim, num_layers=num_layers, embed_dim=embed_dim, num_actions=num_actions,
        market_dim=md,
    )
    model = KnCls().to(device_obj)

    n_bars = len(market_features)
    # Build fixed-length sequences with same size for batching
    seqs = []
    for start in range(0, n_bars - sequence_length, sequence_length // 2):
        end = min(start + sequence_length, n_bars)
        if end - start >= sequence_length // 2:
            seqs.append((start, end))

    n_seqs = len(seqs)
    train_seqs = seqs[:int(n_seqs * (1 - val_ratio))]
    val_seqs = seqs[int(n_seqs * (1 - val_ratio)):]

    print(f"  Sequences: {n_seqs} (train={len(train_seqs)} val={len(val_seqs)})", flush=True)

    cw = _normalize_action_class_weights(class_weights, num_actions)
    action_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device_obj)
        if cw else None
    )
    if cw:
        print(f"  action class_weights={cw}", flush=True)
    reg_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    # Soft-target loss for return-derived action probabilities
    soft_action_loss_fn = nn.KLDivLoss(reduction="batchmean")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    stale = 0

    for ep in range(epochs):
        model.train()
        train_total = 0.0
        n_batches = 0

        # Shuffle sequence order each epoch
        perm = torch.randperm(len(train_seqs))
        shuffled = [train_seqs[i] for i in perm.tolist()]

        for b in range(0, len(shuffled), batch_size):
            batch_seqs = shuffled[b:b + batch_size]
            B = len(batch_seqs)

            # Pack sequences into batched tensors: (S, B, dim)
            # All sequences have the same length within a batch
            S = sequence_length
            mf_batch = torch.zeros(S, B, md, device=device_obj, dtype=torch.float32)
            ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)

            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s
                mf_batch[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)

            # Encode market + position
            m_enc = model.market_encoder(mf_batch)   # (S, B, hidden)
            p_enc = model.pos_encoder(ps_batch)       # (S, B, hidden//2)
            combined = torch.cat([m_enc, p_enc], dim=-1)  # (S, B, hidden+hidden//2)

            # Full batch GRU forward
            gru_out, _ = model.gru(combined)  # (S, B, hidden)

            # Heads on full batch
            batch_act_logits = model.action_head(gru_out)      # (S, B, 6)
            batch_pos_size = torch.sigmoid(model.size_head(gru_out))   # (S, B, 1)
            batch_sl = torch.sigmoid(model.sl_head(gru_out)) * 3.0 + 0.5
            batch_tp = torch.sigmoid(model.tp_head(gru_out)) * 5.0 + 1.0
            batch_trade = model.trade_head(gru_out)  # (S, B, 1)

            # Scenario prediction head (auxiliary task)
            batch_scenarios = model.scenario_head(gru_out)  # (S, B, 64)

            # Multi-task loss
            loss = torch.tensor(0.0, device=device_obj)
            total_bars_in_batch = 0

            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s  # actual sequence length (may be < S)
                act_logits = batch_act_logits[:sl, i]  # (sl, 6)
                pos_size = batch_pos_size[:sl, i]       # (sl, 1)
                sl_mult = batch_sl[:sl, i]
                tp_mult = batch_tp[:sl, i]
                trade = batch_trade[:sl, i]
                total_bars_in_batch += sl

                if "action" in targets:
                    ta = torch.tensor(targets["action"][s:e], dtype=torch.long, device=device_obj)
                    loss = loss + action_loss_fn(act_logits, ta)

                # Soft action targets (return-derived probabilities) — richer signal
                if "action_probs" in targets:
                    tp = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32, device=device_obj)
                    loss = loss + 1.2 * soft_action_loss_fn(
                        torch.log_softmax(act_logits[:, :3], dim=-1), tp)

                if "position_size" in targets:
                    ts = torch.tensor(targets["position_size"][s:e], dtype=torch.float32,
                                      device=device_obj).unsqueeze(1)
                    loss = loss + 0.5 * reg_loss_fn(pos_size, ts)

                if "sl_atr_mult" in targets:
                    tsl = torch.tensor(targets["sl_atr_mult"][s:e], dtype=torch.float32,
                                       device=device_obj).unsqueeze(1)
                    loss = loss + 0.3 * reg_loss_fn(sl_mult, tsl)

                if "tp_atr_mult" in targets:
                    ttp = torch.tensor(targets["tp_atr_mult"][s:e], dtype=torch.float32,
                                       device=device_obj).unsqueeze(1)
                    loss = loss + 0.3 * reg_loss_fn(tp_mult, ttp)

                if "should_trade" in targets:
                    ttr = torch.tensor(targets["should_trade"][s:e], dtype=torch.float32,
                                       device=device_obj).unsqueeze(1)
                    loss = loss + 0.2 * bce_loss_fn(trade, ttr)

                # Scenario prediction loss (auxiliary: predict future delta_price etc.)
                if "scenarios" in targets:
                    tsc = torch.tensor(targets["scenarios"][s:e], dtype=torch.float32,
                                       device=device_obj)
                    loss = loss + 0.3 * reg_loss_fn(batch_scenarios[:sl, i], tsc)

            loss = loss / max(B, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_total += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train = train_total / max(n_batches, 1)

        # Validation
        model.eval()
        val_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for b in range(0, len(val_seqs), batch_size):
                batch_seqs = val_seqs[b:b + batch_size]
                B = len(batch_seqs)
                S = sequence_length
                mf_batch = torch.zeros(S, B, md, device=device_obj, dtype=torch.float32)
                ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)

                for i, (s, e) in enumerate(batch_seqs):
                    sl = e - s
                    mf_batch[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                    ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)

                m_enc = model.market_encoder(mf_batch)
                p_enc = model.pos_encoder(ps_batch)
                gru_out, _ = model.gru(torch.cat([m_enc, p_enc], dim=-1))
                batch_act = model.action_head(gru_out)

                vloss = torch.tensor(0.0, device=device_obj)
                for i, (s, e) in enumerate(batch_seqs):
                    sl = e - s
                    if "action" in targets:
                        ta = torch.tensor(targets["action"][s:e], dtype=torch.long, device=device_obj)
                        vloss = vloss + action_loss_fn(batch_act[:sl, i], ta)
                    if "action_probs" in targets:
                        tp = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32, device=device_obj)
                        vloss = vloss + 0.5 * soft_action_loss_fn(
                            torch.log_softmax(batch_act[:sl, i, :3], dim=-1), tp)
                vloss = vloss / max(B, 1)
                val_total += float(vloss.item())
                n_val_batches += 1

        avg_val = val_total / max(n_val_batches, 1)

        if ep % 10 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1:3d}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}", flush=True)

        if avg_val < best_loss - 0.0001:
            best_loss = avg_val
            stale = 0
            torch.save(model.state_dict(), out)
            meta = {
                "hidden_dim": hidden_dim, "num_layers": num_layers,
                "embed_dim": embed_dim, "num_actions": num_actions, "market_dim": md, "pos_dim": 6,
                "architecture": "kn2_v16" if md == 65 else "kn2_legacy",
                "val_loss": avg_val,
                "class_weights": cw,
                "scenario_trained": "scenarios" in targets,
                "scenario_horizons": SCENARIO_HORIZONS if "scenarios" in targets else [],
            }
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {ep+1}", flush=True)
                break

    return {"val_loss": best_loss, "model_path": str(out)}


# ── 场景预测标签生成 ──

SCENARIO_HORIZONS = [1, 2, 4, 8, 12, 16, 24, 48]
SCENARIO_PARAMS_PER = 8   # delta_price, delta_vol, touch_sl, touch_tp, +4 reserved


def generate_scenario_labels(
    data: "pd.DataFrame",
    *,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    max_hold_bars: int = 48,
) -> np.ndarray:
    """Generate scenario training targets: (n_bars, 64).

    For each bar t and each scenario i (mapping to horizon h_i):
      Param 0: delta_price = (close[t+h_i] - close[t]) / atr[t]
      Param 1: delta_vol   = (vol[t+h_i] - vol[t]) / max(vol[t], 1e-8)
      Param 2: touch_sl    = 1.0 if SL hit in [t+1, t+h_i], else 0
      Param 3: touch_tp    = 1.0 if TP hit in [t+1, t+h_i], else 0
      Param 4-7: 0.0 (reserved)
    """
    import pandas as pd

    n = len(data)
    close = data["close"].values.astype(np.float64)
    high = data["high"].values.astype(np.float64) if "high" in data.columns else close
    low = data["low"].values.astype(np.float64) if "low" in data.columns else close
    vol = data["volume"].values.astype(np.float64) if "volume" in data.columns else np.ones(n)

    atr_raw = data["atr"].values.astype(np.float64) if "atr" in data.columns else np.full(n, close[0] * 0.001)
    atr = np.maximum(atr_raw, close * 0.0005)

    num_h = len(SCENARIO_HORIZONS)
    labels = np.zeros((n, num_h * SCENARIO_PARAMS_PER), dtype=np.float32)

    print(f"  Generating scenario labels for {n:,} bars × {num_h} horizons...", flush=True)

    for t in range(n):
        entry = close[t]
        a = atr[t]

        for i, h in enumerate(SCENARIO_HORIZONS):
            fwd = min(t + h, n - 1)
            off = i * SCENARIO_PARAMS_PER

            # delta_price (normalized by ATR)
            labels[t, off + 0] = (close[fwd] - entry) / a

            # delta_vol
            labels[t, off + 1] = (
                (vol[fwd] - vol[t]) / max(vol[t], 1e-8) if vol[t] > 1e-8 else 0.0
            )

            # touch_sl / touch_tp via barrier check
            fwd_end = min(t + h + 1, n)
            hi = np.max(high[t + 1:fwd_end]) if fwd_end > t + 1 else entry
            lo = np.min(low[t + 1:fwd_end]) if fwd_end > t + 1 else entry

            upper_tp = entry + tp_atr_mult * a
            lower_sl = entry - sl_atr_mult * a
            lower_tp = entry - tp_atr_mult * a
            upper_sl = entry + sl_atr_mult * a

            labels[t, off + 2] = 1.0 if (lo <= lower_sl or hi >= upper_sl) else 0.0
            labels[t, off + 3] = 1.0 if (hi >= upper_tp or lo <= lower_tp) else 0.0

        if t % 50000 == 0 and t > 0:
            print(f"    {t:,}/{n:,}  mean_delta_price={labels[:t, 0].mean():.4f}", flush=True)

    print(f"  Done.  mean_delta_price(h1)={labels[:n, 0].mean():.4f}  "
          f"max_abs_delta={np.max(np.abs(labels[:n, :])):.2f}", flush=True)
    return labels
