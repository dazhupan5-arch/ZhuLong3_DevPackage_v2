"""交易智能体：结构 + 知识网络 + 认知引擎 + PPO 决策。

决策链路：
  M5 K线 → StructureAnalyzer → KnowledgeNet → CausalInference
         → CognitionEngine（市场叙事 + 因果链 + 信号交叉验证 + 信心校准 + 风险评估）
         → StateBuilder → PPO → 动作过滤器 → draw_signal
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhulong.agent.agent_scheduler import AgentScheduler
from zhulong.agent.causal_inference import CausalInference, fuse_knowledge_with_causal
from zhulong.agent.cognition import CognitionEngine
from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.kn2_location_labels import compute_pos_in_range, evaluate_structure_entry_gate
from zhulong.agent.rl_agent import RlAgent, resolve_knowledge_paths, resolve_rl_model_path
from zhulong.agent.state_builder import (
    StateBuilder,
    directional_confidence,
    gate_action_by_cognition,
    primary_direction_from_probs,
)
from zhulong.agent.structure_analyzer import StructureAnalyzer
from zhulong.agent.trader_memory import TraderMemory
from zhulong.strategies.indicators import atr_series
from zhulong.utils.paths import resolve_bundled_data_path, resolve_writable_data_path

logger = logging.getLogger(__name__)

ACTION_NAMES = ["hold", "long", "short", "short_50", "short_100", "close"]


class TradingAgent:
    """实盘 tick：输出 draw_signal 载荷，不直接 OrderSend。"""

    def __init__(self, config: dict[str, Any], root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.use_rl = bool(config.get("use_rl", False))
        self.primary_symbol = str(config.get("primary_symbol", "XAUUSD")).upper()
        self.signal_expiry = int(config.get("signal_expiry_minutes", 240))

        ri = config.get("rl_inference") or {}
        self.rl_min_confidence = float(ri.get("min_confidence_for_trade", 0.65))
        self.rl_action_threshold = float(ri.get("action_threshold", 0.6))
        self.rl_max_daily_trades = int(ri.get("max_daily_trades", 3))
        self.rl_risk_per_trade = float(ri.get("risk_per_trade", 0.005))
        self._daily_trade_counts: dict[str, int] = {}
        self._daily_trade_date = ""

        sa_cfg = config.get("structure_analyzer") or {}
        self.structure = StructureAnalyzer(sa_cfg)

        arch = config.get("architecture") or {}
        self.arch_version = str(arch.get("version", "legacy"))
        self.structure_service = None
        self.horizon: Any = None
        self.trader_mind: Any = None
        if self.arch_version == "v16":
            from zhulong.agent.horizon_predictor import HorizonPredictor
            from zhulong.agent.structure_service import StructureService
            from zhulong.agent.trader_mind import TraderMind

            self.structure_service = StructureService(sa_cfg)
            self.horizon = HorizonPredictor(self.root, config)
            self.trader_mind = TraderMind(config)
            logger.info("Architecture v16: Structure → Horizon → TraderMind")

        eg = config.get("execution_gates") or {}
        self.structure_location_gate = bool(eg.get("structure_location_gate", True))
        self.block_ranging_conflict = bool(eg.get("block_ranging_conflict", True))
        self.horizon_lock_direction = bool(eg.get("horizon_lock_direction", False))

        kn2_cfg = config.get("kn2") or {}
        self.kn2_enabled = bool(kn2_cfg.get("enabled", False))
        self.kn2_shadow = bool(kn2_cfg.get("shadow_mode", False))
        self.kn2_min_confidence = float(kn2_cfg.get("min_confidence", 0.48))
        self._kn2 = None
        if self.arch_version == "v16" and (self.kn2_enabled or self.kn2_shadow):
            self._load_kn2(kn2_cfg)

        rl_cfg = config.get("rl") or {}
        self.rl_deterministic = bool((rl_cfg.get("inference") or {}).get("deterministic", True))
        self._knowledge: KnowledgeNetInference | None = None
        self._rl: RlAgent | None = None
        self.scheduler = None
        self._load_models_for(self.primary_symbol)

        env_cfg = config.get("trading_env") or {}
        self.env_cfg = env_cfg
        self.sl_mult = float(env_cfg.get("stop_loss_atr_mult", 1.2))
        self.tp_mult = float(env_cfg.get("take_profit_atr_mult", 2.0))

        ml_cfg = config.get("meta_learning") or {}
        if ml_cfg.get("enabled", True):
            self.scheduler = AgentScheduler(config, root=self.root)

        self._apply_symbol_context(self.primary_symbol, initial=True)

        mem_cfg = config.get("trader_memory") or {}
        self.memory = TraderMemory(int(mem_cfg.get("max_len", 20)))
        self._state_path = self._resolve_state_file(self.primary_symbol)
        self._load_persisted_state()

        self._last_bar: dict[str, str] = {}
        self._position_hint: dict[str, float] = {}
        self._open_trajectory: list[dict] = []

        causal_cfg = config.get("causal") or {}
        self.causal_enabled = bool(causal_cfg.get("enabled", True))
        self.causal_fusion_weight = float(causal_cfg.get("fusion_weight", 0.3))

        if self.scheduler is not None and self._rl_model is not None:
            self.scheduler.attach_policy(self._rl)

    def _load_kn2(self, kn2_cfg: dict[str, Any]) -> None:
        rel = str(kn2_cfg.get("model_path", "models/kn2_trader_v16.pth"))
        model_path = resolve_bundled_data_path(rel)
        if not model_path.is_file():
            logger.warning("KN2 模型未找到 %s（shadow/live 跳过）", model_path)
            return
        try:
            from zhulong.agent.knowledge_net_kn2 import KN2Inference

            self._kn2 = KN2Inference(model_path, market_dim=65)
            if self._kn2.is_ready:
                logger.info("KN2 已加载 enabled=%s shadow=%s", self.kn2_enabled, self.kn2_shadow)
            else:
                self._kn2 = None
        except Exception as ex:
            logger.warning("KN2 加载失败: %s", ex)
            self._kn2 = None

    def _apply_v16_execution_gates(
        self,
        thought: Any,
        v16_plan: Any,
        cognition_dir: str,
        struct: np.ndarray,
        m5: pd.DataFrame,
        decision_idx: int,
    ) -> None:
        """合并 TraderMind / 认知 / 结构位置门控。"""
        plan_ok = bool(v16_plan.should_trade) if v16_plan is not None else True
        cog_ok = bool(thought.should_trade)
        merged = plan_ok and cog_ok

        if self.block_ranging_conflict and thought.regime in ("ranging", "choppy"):
            conflicts = thought.conflicts or []
            if cognition_dir in ("long", "short") and any("震荡市" in str(c) for c in conflicts):
                merged = False
                thought.risk_warnings = list(thought.risk_warnings or []) + ["震荡市方向冲突 → 不入场"]

        if merged and self.structure_location_gate and cognition_dir in ("long", "short"):
            closes = m5["close"].values.astype(np.float32)
            pos_arr = compute_pos_in_range(closes)
            pos = float(pos_arr[decision_idx]) if decision_idx < len(pos_arr) else 0.5
            ok, reason = evaluate_structure_entry_gate(
                np.asarray(struct, dtype=np.float32),
                pos,
                str(thought.regime or ""),
                cognition_dir,
            )
            if not ok:
                merged = False
                thought.risk_warnings = list(thought.risk_warnings or []) + [reason]

        thought.should_trade = merged
        if v16_plan is not None and not merged and v16_plan.block_reason:
            thought.risk_warnings = list(thought.risk_warnings or []) + [str(v16_plan.block_reason)]

    def _symbol_cfg(self, symbol: str) -> dict[str, Any]:
        return (self.config.get("symbols") or {}).get(symbol.strip().upper()) or {}

    def _resolve_state_scaler(self, symbol: str) -> Path:
        sym_cfg = self._symbol_cfg(symbol)
        sym_key = symbol.strip().lower()
        rel = sym_cfg.get("state_scaler_path") or f"data/agent_state_scaler_{sym_key}.json"
        scaler = resolve_bundled_data_path(rel)
        if not scaler.is_file():
            fallback = self.config.get("state_scaler_path", "data/agent_state_scaler.json")
            scaler = resolve_bundled_data_path(fallback)
        return scaler

    def _resolve_state_file(self, symbol: str) -> Path:
        sym_cfg = self._symbol_cfg(symbol)
        rel = sym_cfg.get("state_file") or self.config.get("state_file", "data/agent_state.json")
        return resolve_writable_data_path(rel)

    def _apply_symbol_context(self, symbol: str, *, initial: bool = False) -> None:
        sym = symbol.strip().upper()
        if not initial:
            self._load_models_for(sym)
        scaler = self._resolve_state_scaler(sym)
        self.state_builder = StateBuilder(scaler if scaler.is_file() else None)
        if not initial:
            self._state_path = self._resolve_state_file(sym)
            self._load_persisted_state()
        causal_cfg = self.config.get("causal") or {}
        coef_path = causal_cfg.get("coef_path", "models/causal_coef.pkl")
        self.causal = CausalInference(self._resolve(coef_path), symbol=sym)
        cog_cfg = dict(self.config.get("cognition") or {})
        cog_cfg.update(self._symbol_cfg(sym).get("cognition") or {})
        cog_cfg["symbol"] = sym
        self.cognition = CognitionEngine({**self.config, "cognition": cog_cfg})
        sched = getattr(self, "scheduler", None)
        if sched is not None and self._rl_model is not None:
            sched.attach_policy(self._rl)

    def _resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _load_models_for(self, symbol: str) -> None:
        kn_path, kn_scaler = resolve_knowledge_paths(symbol, self.config, self.root)
        self._knowledge = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
        if self.arch_version == "v16":
            if not self.horizon or not self.horizon.is_ready:
                if not self._knowledge.is_ready:
                    raise RuntimeError(
                        f"V16 Horizon 未就绪: model={kn_path} scaler={kn_scaler} "
                        f"(需要 models/horizon_v16.onnx + onnxruntime)"
                    )
        elif not self._knowledge.is_ready:
            raise RuntimeError(
                f"KnowledgeNet 未就绪: model={kn_path} scaler={kn_scaler} "
                f"(需要 models/knowledge_net.onnx + onnxruntime)"
            )
        rl_path = resolve_rl_model_path(symbol, self.config, self.root)
        self._rl = RlAgent(rl_path, deterministic=self.rl_deterministic, symbol=symbol)

    @property
    def knowledge(self) -> KnowledgeNetInference:
        assert self._knowledge is not None
        return self._knowledge

    @property
    def _rl_model(self) -> RlAgent | None:
        return self._rl if self._rl and self._rl.is_ready else None

    @staticmethod
    def _build_v15_features(m5: pd.DataFrame) -> np.ndarray:
        """从 M5 构建 V15 76 维特征（KN1 V15 蒸馏）。"""
        from zhulong.training.lgb.features_v15 import FEATURE_COLUMNS_V15, compute_features_v15

        feats = compute_features_v15(m5)
        if feats.empty:
            return np.zeros(len(FEATURE_COLUMNS_V15), dtype=np.float32)
        cols = list(FEATURE_COLUMNS_V15)
        return feats.iloc[-1][cols].to_numpy(dtype=np.float32)

    @staticmethod
    def _build_v14_features(m5: pd.DataFrame) -> np.ndarray:
        """从 M5 数据构建 68 维 V14 特征供 KnowledgeNet 推理。"""
        from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
        feats = compute_features(m5, include_mtf=True, include_reversal=True)
        if feats.empty:
            return np.zeros(len(FEATURE_COLUMNS_LGB_V13), dtype=np.float32)
        cols = list(FEATURE_COLUMNS_LGB_V13)
        return feats.iloc[-1][cols].to_numpy(dtype=np.float32)

    def _build_knowledge_features(self, m5: pd.DataFrame, struct: np.ndarray) -> np.ndarray:
        """XAU V15=76维 / V14=68维；USOIL 等用 30 维结构特征。"""
        kn = self.knowledge
        input_dim = int(getattr(kn, "input_dim", 68))
        # 必须用 model input_dim 判定版本；keep_cols.max()==67 的 V14 模型不能走 V15 特征
        if input_dim > 68:
            return self._build_v15_features(m5)
        if input_dim <= 30:
            return np.asarray(struct[:30], dtype=np.float32)
        return self._build_v14_features(m5)

    def _load_persisted_state(self) -> None:
        p = Path(self._state_path)
        if not p.is_file():
            return
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
            self.memory.load_list(blob.get("trades"))
            self._position_hint = blob.get("position_hint") or {}
            cog = blob.get("cognition")
            if cog:
                self.cognition.import_state(cog)
        except Exception as ex:
            logger.warning("智能体状态加载失败: %s", ex)

    def _save_state(self) -> None:
        try:
            p = Path(self._state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "trades": self.memory.to_list(),
                    "position_hint": self._position_hint,
                    "cognition": self.cognition.export_state(),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "primary_symbol": self.primary_symbol,
                },
                ensure_ascii=False,
            )
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(p)
        except Exception as ex:
            logger.warning("智能体状态保存失败 path=%s: %s", self._state_path, ex)

    def set_primary_symbol(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        if sym == self.primary_symbol:
            return
        self.primary_symbol = sym
        self._apply_symbol_context(sym)
        logger.info("智能体已切换主品种 → %s (KN ready=%s, RL ready=%s)",
                    sym, self.knowledge.is_ready, self._rl_model is not None)

    def record_closed_trade(self, symbol: str, pnl_r: float) -> None:
        self.memory.add_trade(pnl_r, datetime.now(timezone.utc).isoformat())
        # 反馈给认知引擎
        self.cognition.record_outcome(pnl_r > 0)
        if self.scheduler is not None and self._open_trajectory:
            self.scheduler.on_trade_closed(self._open_trajectory, pnl_r)
            self._open_trajectory = []
        self._save_state()

    @staticmethod
    def _directional_ai_prices(
        thought: Any,
        direction: str,
        entry: float,
        atr: float,
        sl_mult: float,
        tp_mult: float,
    ) -> tuple[float, float]:
        """按实际信号方向取认知 SL/TP；若与方向不符则按 ATR 重算。"""
        ai_sl = float(getattr(thought, "ai_sl_price", 0.0) or 0.0) if thought is not None else 0.0
        ai_tp = float(getattr(thought, "ai_tp_price", 0.0) or 0.0) if thought is not None else 0.0
        if direction == "buy":
            if ai_sl <= 0 or ai_sl >= entry:
                ai_sl = entry - sl_mult * atr
            if ai_tp <= 0 or ai_tp <= entry:
                ai_tp = entry + tp_mult * atr
        else:
            if ai_sl <= 0 or ai_sl <= entry:
                ai_sl = entry + sl_mult * atr
            if ai_tp <= 0 or ai_tp >= entry:
                ai_tp = entry - tp_mult * atr
        return ai_sl, ai_tp

    def _action_to_signal(
        self,
        action: int,
        symbol: str,
        close: float,
        atr: float,
        confidence: float,
        probs: np.ndarray | None,
        causal_pred: float = 0.0,
        thought: Any = None,
        bar_high: float | None = None,
        bar_low: float | None = None,
    ) -> dict[str, Any] | None:
        sym_cfg = (self.config.get("symbols") or {}).get(symbol, {})
        broker = sym_cfg.get("broker_symbol") or symbol

        if action == 0 or action == 5:
            return {
                "strategy": "rl_agent",
                "symbol": symbol,
                "direction": "flat",
                "confidence": confidence,
                "entry": close,
                "sl": 0.0,
                "tp": 0.0,
                "signal_id": "",
                "reject_reason": "hold" if action == 0 else "close_only",
                "broker_symbol": broker,
                "metadata": {"action": ACTION_NAMES[action], "rl_model": self._rl_model is not None},
            }

        direction = "buy" if action == 1 else "sell"
        if thought is not None and getattr(thought, "ai_entry_price", 0.0) > 0:
            entry = float(thought.ai_entry_price)
        elif direction == "buy":
            entry = min(close, (bar_low + close) / 2) if bar_low else close
        else:
            entry = max(close, (bar_high + close) / 2) if bar_high else close

        if thought is not None and hasattr(thought, "sl_mult") and thought.sl_mult > 0:
            sl_mult, tp_mult = thought.sl_mult, thought.tp_mult
        else:
            sl_mult, tp_mult = self.sl_mult, self.tp_mult

        if direction == "buy":
            sl = entry - sl_mult * atr
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            tp = entry - tp_mult * atr

        sl, tp = self._directional_ai_prices(thought, direction, entry, atr, sl_mult, tp_mult)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        sid = f"agent_{ts}_{symbol}_{direction}_{uuid.uuid4().hex[:6]}"
        meta = {
            "action": ACTION_NAMES[action],
            "action_id": action,
            "rl_model": self._rl_model is not None,
        }
        if probs is not None and probs.size >= 3:
            meta["knowledge_probs"] = [float(x) for x in probs.reshape(-1)[:3]]
        meta["causal_pred"] = float(causal_pred)
        meta["comment"] = "RL_Agent"

        return {
            "strategy": "rl_agent",
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "signal_id": sid,
            "reject_reason": "",
            "broker_symbol": broker,
            "metadata": meta,
        }

    def _resolve_rl_position_hint(self, symbol: str, account: dict[str, Any]) -> float:
        """RL 状态向量中的持仓维：以 C# 托管持仓为准，无仓时不沿用磁盘 position_hint。"""
        for pos in account.get("_positions") or []:
            if not isinstance(pos, dict):
                continue
            ps = str(pos.get("symbol") or "")
            if ps.upper() != symbol.upper():
                continue
            direction = str(pos.get("direction") or "")
            if direction == "buy":
                return 1.0
            if direction == "sell":
                return -1.0
            return 0.0
        return 0.0

    def on_bar(
        self,
        symbol: str,
        m5_by_symbol: dict[str, pd.DataFrame],
        account: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        m5 = m5_by_symbol.get(symbol)
        if m5 is None or m5.empty:
            raise RuntimeError(f"no_m5:{symbol}")

        acct = account or {}
        includes_forming = bool(acct.get("_m5_includes_forming", True))
        # C# TryExportAgentM5Bars 已去掉形成中 K 线时，最后一根即为决策 bar
        decision_idx = -2 if includes_forming and len(m5) >= 2 else -1
        bar_time = m5.index[decision_idx]
        bar_key = str(bar_time)
        decision_unix = int(acct.get("_decision_bar_unix") or 0)
        if decision_unix > 0:
            try:
                target = pd.to_datetime(decision_unix, unit="s", utc=True)
                if target in m5.index:
                    decision_idx = int(m5.index.get_loc(target))
                    bar_time = m5.index[decision_idx]
                    bar_key = str(bar_time)
            except Exception:
                pass

        if self._last_bar.get(symbol) == bar_key:
            return []
        self._last_bar[symbol] = bar_key

        if symbol.upper() != self.primary_symbol:
            self.set_primary_symbol(symbol.upper())

        close = float(m5["close"].iloc[decision_idx])
        high = float(m5["high"].iloc[decision_idx])
        low = float(m5["low"].iloc[decision_idx])
        atr_s = atr_series(m5)
        atr = float(atr_s.iloc[decision_idx]) if not pd.isna(atr_s.iloc[decision_idx]) else close * 0.001
        cons_losses = self.memory.get_consecutive_losses()

        mtf = {symbol: m5}
        struct = self.structure.compute_latest({"M5": m5})

        v16_plan = None
        v16_forecast = None
        v16_snapshot = None
        market_feat_kn2 = None
        if self.arch_version == "v16" and self.structure_service and self.horizon and self.trader_mind:
            m5s = m5.sort_index()
            loc = m5s.index.get_loc(bar_time)
            if isinstance(loc, slice):
                loc = len(m5s) - 1
            v16_snapshot = self.structure_service.snapshot_from_row(m5s, int(loc))
            v16_forecast = self.horizon.predict(v16_snapshot)
            cons_losses = self.memory.get_consecutive_losses()
            v16_plan = self.trader_mind.plan(
                v16_forecast,
                v16_snapshot,
                close=close,
                atr=atr,
                consecutive_losses=cons_losses,
                regime=str(v16_snapshot.zigzag_phase or ""),
            )
            kn_row = np.asarray(v16_snapshot.vector, dtype=np.float32)
            prob_row = np.array(v16_forecast.to_kn_probs(), dtype=np.float32)
            emb = np.zeros(32, dtype=np.float32)
            if self.horizon.is_ready and self.horizon._kn is not None:
                _, emb = self.horizon._kn.predict(kn_row.reshape(1, -1))
                emb = emb.reshape(-1) if emb is not None else emb
            market_feat_kn2 = np.concatenate(
                [kn_row.reshape(-1)[:30], prob_row.reshape(-1)[:3], emb.reshape(-1)[:32]]
            ).astype(np.float32)
        else:
            kn_row = self._build_knowledge_features(m5, struct)
            probs, emb = self.knowledge.predict(kn_row.reshape(1, -1))
            prob_row = probs[0] if probs.ndim > 1 else probs

        causal_pred = 0.0
        raw_causal_pred = 0.0
        if self.causal_enabled and self.arch_version != "v16":
            shock = self.causal.macro_shock_from_bar(struct)
            raw_causal_pred = self.causal.predict_price_change(shock)
            causal_pred = raw_causal_pred
            if self.causal_fusion_weight > 0:
                fused = fuse_knowledge_with_causal(
                    prob_row,
                    causal_pred,
                    weight_knowledge=1.0 - self.causal_fusion_weight,
                    weight_causal=self.causal_fusion_weight,
                )
                prob_row = fused[0] if fused.ndim > 1 else fused

        # ================================================================
        # 认知引擎：像交易员一样思考
        # ================================================================
        volume_val = float(m5["volume"].iloc[decision_idx]) if "volume" in m5.columns else 0.0
        now = datetime.now(timezone.utc)
        time_of_day = (now.hour + now.minute / 60.0, now.weekday())

        # 子进程 tick：从 M5 历史重建语境，修复 regime=unknown
        self.cognition.rebuild_context_from_m5(m5)

        tick_bid = tick_ask = 0.0
        ticks = acct.get("_ticks") or {}
        sym_tick = ticks.get(symbol) or ticks.get(symbol.upper()) or {}
        if isinstance(sym_tick, dict):
            tick_bid = float(sym_tick.get("bid") or 0.0)
            tick_ask = float(sym_tick.get("ask") or 0.0)

        position_ctx = None
        for pos in acct.get("_positions") or []:
            if not isinstance(pos, dict):
                continue
            ps = str(pos.get("symbol") or "")
            if ps.upper() == symbol.upper():
                position_ctx = pos
                break

        lock_dir = None
        if self.horizon_lock_direction and v16_forecast is not None:
            lock_dir = v16_forecast.direction

        thought = self.cognition.process(
            struct_features=struct,
            knowledge_probs=prob_row,
            causal_pred=raw_causal_pred,
            close=close,
            atr=atr,
            volume=volume_val,
            bar_timestamp=bar_key,
            consecutive_losses=cons_losses,
            time_of_day=time_of_day,
            tick_bid=tick_bid,
            tick_ask=tick_ask,
            position_ctx=position_ctx,
            lock_forecast_direction=lock_dir,
        )

        kn2_dec: dict[str, Any] | None = None
        horizon_min_conf = float(self.horizon.min_confidence) if self.horizon else 0.48
        if self._kn2 is not None and market_feat_kn2 is not None and self._kn2.is_ready:
            from zhulong.agent.knowledge_net_kn2 import encode_position_state

            pos_state = encode_position_state(
                direction=float(position_ctx.get("direction_sign", 0) if position_ctx else 0),
            )
            try:
                kn2_dec = self._kn2.predict(market_feat_kn2, pos_state)
                logger.info(
                    "[KN2%s] %s should_trade=%s action=%s conf=%.3f sl=%.2f tp=%.2f",
                    "·shadow" if self.kn2_shadow and not self.kn2_enabled else "",
                    symbol,
                    kn2_dec.get("should_trade"),
                    kn2_dec.get("action_name", kn2_dec.get("action")),
                    float(kn2_dec.get("confidence", 0)),
                    float(kn2_dec.get("sl_atr_mult", 0)),
                    float(kn2_dec.get("tp_atr_mult", 0)),
                )
            except Exception as ex:
                logger.warning("KN2 predict 失败: %s", ex)

        if v16_forecast is not None and v16_plan is not None:
            prob_row = np.array(v16_forecast.to_kn_probs(), dtype=np.float32)
            cognition_dir = v16_forecast.direction
            self._apply_v16_execution_gates(
                thought, v16_plan, cognition_dir, struct, m5, decision_idx
            )
            if v16_plan.sl_price > 0:
                thought.ai_sl_price = v16_plan.sl_price
            if v16_plan.tp_price > 0:
                thought.ai_tp_price = v16_plan.tp_price
            if kn2_dec and self.kn2_enabled:
                if not kn2_dec.get("should_trade"):
                    thought.should_trade = False
                    thought.risk_warnings = list(thought.risk_warnings or []) + ["KN2否决入场"]
                elif float(kn2_dec.get("confidence", 0)) < self.kn2_min_confidence:
                    thought.should_trade = False
                    thought.risk_warnings = list(thought.risk_warnings or []) + ["KN2置信不足"]
                else:
                    kn2_sl = float(kn2_dec.get("sl_atr_mult", 0))
                    kn2_tp = float(kn2_dec.get("tp_atr_mult", 0))
                    if kn2_sl > 0 and cognition_dir == "long":
                        thought.ai_sl_price = close - kn2_sl * atr
                    elif kn2_sl > 0 and cognition_dir == "short":
                        thought.ai_sl_price = close + kn2_sl * atr
                    if kn2_tp > 0 and cognition_dir == "long":
                        thought.ai_tp_price = close + kn2_tp * atr
                    elif kn2_tp > 0 and cognition_dir == "short":
                        thought.ai_tp_price = close - kn2_tp * atr
            instant_dir = cognition_dir
            smoothed_regime = thought.regime
        else:
            prob_row = np.array(thought.calibrated_probs, dtype=np.float32)
            cognition_dir, smoothed_regime, instant_dir = self.cognition.resolve_sticky_direction(
                thought.calibrated_probs,
                bar_key,
            )
        if instant_dir != cognition_dir:
            logger.info(
                "[认知观点] %s bar=%s 即时=%s 维持=%s 行情=%s→%s",
                symbol,
                bar_key,
                instant_dir,
                cognition_dir,
                thought.regime,
                smoothed_regime,
            )
        cognition_conf = directional_confidence(
            prob_row, 1 if cognition_dir == "long" else 2 if cognition_dir == "short" else 0, thought.confidence
        )

        if self.use_rl and thought.should_trade:
            if cognition_dir == "flat":
                thought.should_trade = False
                thought.risk_warnings = list(thought.risk_warnings or []) + ["认知方向不明确"]
            else:
                dir_conf = directional_confidence(
                    prob_row, 1 if cognition_dir == "long" else 2, thought.confidence
                )
                if dir_conf < self.rl_min_confidence or dir_conf < self.rl_action_threshold:
                    thought.should_trade = False
                    thought.risk_warnings = list(thought.risk_warnings or []) + ["RL置信门槛未达"]
        # ================================================================

        acct.setdefault("initial_balance", float(self.env_cfg.get("initial_balance", 10000)))
        acct.setdefault("balance", acct["initial_balance"])
        acct.setdefault("equity", acct.get("balance"))
        pos_hint = self._resolve_rl_position_hint(symbol, acct)
        acct.setdefault("position", pos_hint)

        cognition_ctx = {
            "calibrated_probs": thought.calibrated_probs,
            "regime": smoothed_regime or thought.regime,
            "confidence": thought.confidence,
            "should_trade": thought.should_trade,
        }
        state = self.state_builder.build(
            struct, emb.reshape(-1), acct, self.memory, bar_time, cognition=cognition_ctx
        )

        rl_raw_action = 0
        rl_raw_name = "hold"
        gate_reason = ""
        if self.use_rl and self._rl_model is not None:
            rl_raw_action, _ = self._rl_model.predict(state)
            rl_raw_name = ACTION_NAMES[rl_raw_action]
            logger.info(
                "[RL意见] %s raw=%s cog=%s cog_conf=%.3f should_trade=%s regime=%s bar=%s pos_hint=%.1f",
                symbol,
                rl_raw_name,
                cognition_dir,
                cognition_conf,
                thought.should_trade,
                thought.regime,
                bar_key,
                pos_hint,
            )
            if not thought.should_trade:
                action = 0
                confidence = cognition_conf
                if rl_raw_action in (1, 2, 3, 4):
                    warnings = [str(w) for w in (thought.risk_warnings or [])]
                    if any("RL置信门槛未达" in w for w in warnings):
                        gate_reason = "rl_confidence_threshold"
                    elif any("置信度过低" in w for w in warnings):
                        gate_reason = "cognition_confidence_low"
                    elif any("风险过高" in w for w in warnings):
                        gate_reason = "risk_too_high"
                    elif any("连亏" in w for w in warnings):
                        gate_reason = "consecutive_losses"
                    elif any("认知方向不明确" in w for w in warnings):
                        gate_reason = "cognition_flat"
                    else:
                        gate_reason = "should_trade_false"
                    if gate_reason:
                        logger.info(
                            "[RL门控] %s PPO=%s 认知=%s → hold (%s)",
                            symbol,
                            rl_raw_name,
                            cognition_dir,
                            gate_reason,
                        )
            else:
                action = rl_raw_action
                confidence = directional_confidence(prob_row, action, cognition_conf)
                action, gate_reason = gate_action_by_cognition(action, cognition_dir)
                if gate_reason:
                    logger.info(
                        "[RL门控] %s PPO=%s 认知=%s → hold (%s)",
                        symbol,
                        rl_raw_name,
                        cognition_dir,
                        gate_reason,
                    )
        else:
            action, confidence = self._heuristic_action(prob_row, struct)
            if not thought.should_trade:
                action = 0
            else:
                action, gate_reason = gate_action_by_cognition(action, cognition_dir)
                confidence = directional_confidence(prob_row, action, max(confidence, cognition_conf))

        action, filter_reason = self._apply_rl_inference_filters(
            action, confidence, prob_row, symbol, bar_key, cognition_dir
        )
        if (
            not filter_reason
            and self.structure_location_gate
            and action in (1, 2)
            and cognition_dir in ("long", "short")
        ):
            closes = m5["close"].values.astype(np.float32)
            pos_arr = compute_pos_in_range(closes)
            pos = float(pos_arr[decision_idx]) if decision_idx < len(pos_arr) else 0.5
            ok, reason = evaluate_structure_entry_gate(
                np.asarray(struct, dtype=np.float32),
                pos,
                str(thought.regime or ""),
                cognition_dir,
            )
            if not ok:
                action = 0
                filter_reason = reason
        if gate_reason and not filter_reason:
            filter_reason = gate_reason

        # 入场 tick 评估使用 RL 动作方向，避免 trade_bias 与 RL 不一致
        if action in (1, 2, 3, 4):
            entry_dir = "buy" if action == 1 else "sell"
            entry_eval = self.cognition._evaluate_entry(
                direction=entry_dir,
                tick_bid=tick_bid,
                tick_ask=tick_ask,
                bar_close=close,
                atr=atr,
                regime=thought.regime,
                ai_sl=0.0,
            )
            if entry_eval.get("should_wait"):
                action = 0
                filter_reason = entry_eval.get("reason") or "entry_wait_tick"
            else:
                ep = float(entry_eval.get("entry_price") or close)
                if ep > 0:
                    thought.ai_entry_price = ep
                ai_sl, ai_tp = self.cognition.sl_tp_for_direction(
                    entry_dir, thought, struct, ep, atr, entry_anchored=True
                )
                if entry_dir == "buy" and ep <= ai_sl:
                    action = 0
                    filter_reason = "入场价低于智能体止损"
                elif entry_dir == "sell" and ep >= ai_sl:
                    action = 0
                    filter_reason = "入场价高于智能体止损"
                else:
                    thought.ai_sl_price = ai_sl
                    thought.ai_tp_price = ai_tp
        # 持仓中：SL/TP 由 tick 移动止损管理，M5 不再重算结构止损（避免逆向放宽）
        if position_ctx and str(position_ctx.get("direction") or "") in ("buy", "sell"):
            thought.ai_sl_price = 0.0
            thought.ai_tp_price = 0.0

        if position_ctx:
            exit_eval = self.cognition.evaluate_exit_for_position(
                thought,
                position_ctx,
                rl_raw_action,
                close,
                atr,
                tick_bid=tick_bid,
                tick_ask=tick_ask,
            )
            thought.exit_assessment = exit_eval["exit_score"]
            thought.exit_reason = exit_eval["exit_reason"]
            thought.exit_reasoning = exit_eval["reasoning"]

        if self.scheduler is not None:
            action = self.scheduler.apply_action_bias(action)
            meta_result = self.scheduler.on_tick()
            if meta_result:
                logger.info("元学习调度: %s", meta_result)

        info = {
            "symbol": symbol,
            "strategy": "rl_agent",
            "state": "AGENT",
            "bar_time": bar_key,
            "close": close,
            "atr": atr,
            "action": ACTION_NAMES[action],
            "action_id": action,
            "rl_raw_action": rl_raw_name,
            "rl_raw_action_id": rl_raw_action,
            "cognition_direction": cognition_dir,
            "cognition_confidence": round(cognition_conf, 4),
            "filter_reason": filter_reason or gate_reason or "",
            "architecture": self.arch_version,
            "horizon_direction": v16_forecast.direction if v16_forecast else "",
            "horizon_confidence": round(float(v16_forecast.confidence), 4) if v16_forecast else 0.0,
            "horizon_min_confidence": round(horizon_min_conf, 4),
            "kn2_should_trade": bool(kn2_dec.get("should_trade")) if kn2_dec else False,
            "kn2_action": (kn2_dec.get("action_name") or str(kn2_dec.get("action", ""))) if kn2_dec else "",
            "kn2_confidence": round(float(kn2_dec.get("confidence", 0)), 4) if kn2_dec else 0.0,
            "kn2_shadow_mode": self.kn2_shadow and not self.kn2_enabled,
            "rl_loaded": self.use_rl and self._rl_model is not None,
            "use_rl": self.use_rl,
            "knowledge_ready": (
                bool(self.horizon and self.horizon.is_ready)
                if self.arch_version == "v16"
                else self.knowledge.is_ready
            ),
            "causal_pred": causal_pred,
            "causal_ready": self.causal.is_ready,
            # 认知引擎输出
            "cognition": {
                "regime": thought.regime,
                "regime_confidence": thought.regime_confidence,
                "narrative": thought.narrative,
                "narrative_events": thought.narrative_events,
                "agreement_score": thought.agreement_score,
                "conflicts": thought.conflicts,
                "confidence": thought.confidence,
                "reasoning_chain": thought.reasoning_chain,
                "risk_score": thought.risk_score,
                "risk_warnings": thought.risk_warnings,
                "position_mult": thought.position_mult,
                "should_trade": thought.should_trade,
            },
        }

        self._open_trajectory.append({"step": bar_key, "state": state.copy(), "action": action, "reward": 0.0})
        sig = self._action_to_signal(action, symbol, close, atr, confidence, prob_row, causal_pred, thought, high, low)
        if filter_reason and sig:
            sig = dict(sig)
            sig["direction"] = "flat"
            sig["reject_reason"] = filter_reason

        # 将认知思维轨迹附到 metadata
        if sig and sig.get("metadata") is not None:
            sig["metadata"]["cognition_thought"] = thought.to_dict()
            sig["metadata"]["cognition_log"] = thought.to_log_line()

        result = {**info, "signal": sig}
        if thought is not None:
            result["exit_assessment"] = round(thought.exit_assessment, 4)
            result["exit_reason"] = thought.exit_reason or ""
            result["ai_sl_price"] = round(thought.ai_sl_price, 5) if thought.ai_sl_price > 0 else 0.0
            result["ai_tp_price"] = round(thought.ai_tp_price, 5) if thought.ai_tp_price > 0 else 0.0
            result["cognition_regime"] = thought.regime
            result["cognition_regime_confidence"] = round(thought.regime_confidence, 4)
            result["cognition_direction"] = cognition_dir
            result["cognition_confidence"] = round(cognition_conf, 4)
        # 思维日志
        logger.info("[THOUGHT] %s", thought.to_log_line())

        if sig and sig.get("direction") in ("buy", "sell"):
            payload = {
                "action": "draw_signal",
                "signal_id": sig["signal_id"],
                "symbol": sig.get("broker_symbol") or symbol,
                "direction": sig["direction"],
                "entry": round(sig["entry"], 5),
                "sl": round(sig["sl"], 5),
                "tp": round(sig["tp"], 5),
                "confidence": round(sig["confidence"], 4),
                "strategy": "rl_agent",
                "market_state": "AGENT",
                "expiry_minutes": self.signal_expiry,
                "meta": sig.get("metadata"),
                # ===== P2-2: AI 退出评估 =====
                "exit_assessment": round(thought.exit_assessment, 4) if thought is not None else 0.0,
                "exit_reason": thought.exit_reason if thought is not None else "",
                "ai_sl_price": round(thought.ai_sl_price, 5) if (thought is not None and thought.ai_sl_price > 0) else 0.0,
                "ai_tp_price": round(thought.ai_tp_price, 5) if (thought is not None and thought.ai_tp_price > 0) else 0.0,
                # ===== 结束 =====
            }
            result["draw_payload"] = payload
            day = bar_key[:10]
            if day != self._daily_trade_date:
                self._daily_trade_date = day
                self._daily_trade_counts = {}
            self._daily_trade_counts[symbol] = self._daily_trade_counts.get(symbol, 0) + 1
            if action == 1:
                self._position_hint[symbol] = 1.0
            elif action == 2:
                self._position_hint[symbol] = -1.0
            elif action in (3, 4):
                self._position_hint[symbol] = -0.5 if action == 3 else -1.0
            elif action in (0, 5):
                self._position_hint[symbol] = 0.0

        self._save_state()
        return [result]

    def _apply_rl_inference_filters(
        self,
        action: int,
        confidence: float,
        probs: np.ndarray,
        symbol: str,
        bar_key: str,
        cognition_dir: str = "flat",
    ) -> tuple[int, str]:
        if action not in (1, 2, 3, 4):
            return action, ""
        if not self.use_rl:
            return action, ""

        day = bar_key[:10]
        if day != self._daily_trade_date:
            self._daily_trade_date = day
            self._daily_trade_counts = {}
        if self._daily_trade_counts.get(symbol, 0) >= self.rl_max_daily_trades:
            return 0, "max_daily_trades"

        trade_conf = directional_confidence(probs, action, confidence)
        if trade_conf < self.rl_min_confidence or trade_conf < self.rl_action_threshold:
            return 0, "low_confidence"

        gated, reason = gate_action_by_cognition(action, cognition_dir)
        if reason:
            return gated, reason

        return action, ""

    @staticmethod
    def _heuristic_action(probs: np.ndarray, struct: np.ndarray) -> tuple[int, float]:
        """无 RL 模型时用知识网络 + 趋势。标签顺序：0=空 1=观望 2=多。"""
        p = probs.reshape(-1)
        if p.size < 3:
            p = np.array([0.34, 0.33, 0.33])
        short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])
        trend = float(struct[0]) if struct.size else 0.0
        if long_p > 0.45 and trend > 0:
            return 1, long_p
        if short_p > 0.45 and trend < 0:
            return 2, short_p
        if long_p > short_p and long_p > flat_p and long_p > 0.42:
            return 1, long_p
        if short_p > long_p and short_p > flat_p and short_p > 0.42:
            return 2, short_p
        return 0, max(long_p, flat_p, short_p)

    def tick_symbols(
        self,
        m5_by_symbol: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
        account: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        primary = self.primary_symbol
        syms = symbols or list(m5_by_symbol.keys())
        if primary not in syms:
            return []
        return self.on_bar(primary, m5_by_symbol, account)
