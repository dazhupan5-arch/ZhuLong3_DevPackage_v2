"""自动调度引擎：SchedulerCore + 策略池 + M5 tick。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from zhulong.utils.paths import resolve_runtime_path
from zhulong.scheduler.scheduler_core import SchedulerCore
from zhulong.scheduler.types import ModelPrediction, SchedulerOutput
from zhulong.strategies.ai_model import AIModelStrategy
from zhulong.strategies.base import StrategyContext, StrategySignal
from zhulong.strategies.grid_system import GridSystem
from zhulong.strategies.spread_hedge import SpreadHedge
from zhulong.strategies.trend_system import TrendSystem

logger = logging.getLogger(__name__)


def load_scheduler_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return json.loads(p.read_text(encoding="utf-8-sig"))


def merge_scheduler_config(base: dict[str, Any], scheduler_path: str | Path | None) -> dict[str, Any]:
    merged = dict(base)
    sched_ref = merged.get("scheduler") or {}
    path = scheduler_path or sched_ref.get("config_path") or "config/config_scheduler.json"
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    enabled = bool(sched_ref.get("enabled", False))
    if p.is_file():
        sched_full = load_scheduler_config(p)
        merged["scheduler_core"] = {
            "weight_allocator": sched_full.get("weight_allocator") or {},
            "state_machine": sched_full.get("state_machine") or {},
            "risk_manager": sched_full.get("risk_manager") or {},
            "vote_margin": sched_full.get("vote_margin", 0.1),
            "min_emit_weight": sched_full.get("min_emit_weight", 0.15),
            "other_strategies": sched_full.get("other_strategies") or {},
        }
        if sched_ref.get("enabled") is None and sched_full.get("scheduler"):
            enabled = bool(sched_full["scheduler"].get("enabled", True))
        elif sched_ref.get("enabled") is None:
            enabled = True
    merged["scheduler_enabled"] = enabled
    return merged


class SchedulerEngine:
    """MultiStrategyEngine 的调度增强版。"""

    def __init__(self, config: dict[str, Any], root: Path | None = None) -> None:
        self.config = config
        self.root = root or Path.cwd()
        self.enabled = bool(config.get("enabled", True))
        self.signal_expiry = int(config.get("signal_expiry_minutes", 240))
        self.macro_silence = bool(config.get("macro_silence", False))

        sched_cfg = config.get("scheduler_core") or {}
        if not sched_cfg and config.get("weight_allocator"):
            sched_cfg = {
                "weight_allocator": config.get("weight_allocator"),
                "state_machine": config.get("state_machine"),
                "risk_manager": config.get("risk_manager"),
                "vote_margin": config.get("vote_margin", 0.1),
                "min_emit_weight": config.get("min_emit_weight", 0.15),
                "other_strategies": config.get("other_strategies"),
            }
        self.scheduler = SchedulerCore(sched_cfg)
        self.primary_symbol = self.scheduler.primary_symbol

        sym_cfg = config.get("symbols") or {}
        self.strategies = {
            "ai_model": AIModelStrategy({"symbols": sym_cfg}, self.root),
            "trend_system": TrendSystem(config.get("trend_system") or sched_cfg.get("other_strategies", {}).get("trend_system") or {}),
            "spread_hedge": SpreadHedge(config.get("spread_hedge") or sched_cfg.get("other_strategies", {}).get("spread_hedge") or {}),
            "grid_system": GridSystem(config.get("grid_system") or sched_cfg.get("other_strategies", {}).get("grid_system") or {}),
        }
        self._last_bar: dict[str, str] = {}
        self._state_path = self._resolve_state_path(config)

        if self._state_path.is_file():
            try:
                blob = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.scheduler.load_blob(blob)
            except Exception as ex:
                logger.warning("调度状态加载失败: %s", ex)

    def set_primary_symbol(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.primary_symbol = sym
        self.scheduler.primary_symbol = sym
        self.scheduler.state_machine.primary_symbol = sym

    def _resolve_state_path(self, config: dict[str, Any]) -> Path:
        rel = (config.get("scheduler") or {}).get("state_file") or "data/scheduler_state.json"
        return resolve_runtime_path(rel, root=self.root)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(self.scheduler.persist_blob(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as ex:
            logger.warning("调度状态保存失败: %s", ex)

    def _context(self, m5_by_symbol: dict[str, pd.DataFrame], bar_time: pd.Timestamp | None) -> StrategyContext:
        return StrategyContext(
            m5_by_symbol=m5_by_symbol,
            config=self.config,
            macro_silence=self.macro_silence,
            bar_time=bar_time,
        )

    @staticmethod
    def _strategy_signal_to_prediction(sig: StrategySignal) -> ModelPrediction:
        d = 1 if sig.direction == "buy" else (-1 if sig.direction == "sell" else 0)
        return ModelPrediction(
            symbol=sig.symbol,
            direction=d,
            confidence=sig.confidence,
            entry=sig.entry,
            sl=sig.sl,
            tp=sig.tp,
            signal_id=sig.signal_id,
            broker_symbol=sig.broker_symbol or sig.symbol,
            reject_reason=sig.reject_reason,
            metadata=sig.metadata,
        )

    @staticmethod
    def _scheduler_output_to_result(out: SchedulerOutput, info: dict[str, Any], expiry: int) -> dict[str, Any]:
        sig_dict = {
            "strategy": out.strategy,
            "symbol": out.symbol,
            "direction": out.direction,
            "confidence": out.confidence,
            "entry": out.entry,
            "sl": out.sl,
            "tp": out.tp,
            "signal_id": out.signal_id,
            "reject_reason": out.reject_reason,
            "metadata": {
                **out.metadata,
                "risk_weight": out.risk_weight,
                "weights": out.weights,
                "market_state": out.market_state,
            },
        }
        result = {**info, "signal": sig_dict, "scheduler": True}
        if out.direction not in ("flat", ""):
            payload = {
                "action": "draw_signal",
                "signal_id": out.signal_id,
                "symbol": out.broker_symbol or out.symbol,
                "direction": out.direction,
                "entry": round(out.entry, 5),
                "sl": round(out.sl, 5),
                "tp": round(out.tp, 5),
                "confidence": round(out.confidence, 4),
                "strategy": out.strategy,
                "market_state": out.market_state,
                "expiry_minutes": expiry,
                "meta": sig_dict.get("metadata"),
            }
            result["draw_payload"] = payload
        return result

    @staticmethod
    def _strategy_result(sig: StrategySignal | None, info: dict[str, Any], expiry: int) -> dict[str, Any]:
        if sig is None:
            return {**info, "signal": None}
        result = {
            **info,
            "signal": {
                "strategy": sig.strategy,
                "symbol": sig.symbol,
                "direction": sig.direction,
                "confidence": sig.confidence,
                "entry": sig.entry,
                "sl": sig.sl,
                "tp": sig.tp,
                "signal_id": sig.signal_id,
                "reject_reason": sig.reject_reason,
                "metadata": sig.metadata,
            },
        }
        if sig.direction != "flat":
            result["draw_payload"] = sig.to_draw_payload(expiry)
        return result

    def on_bar(self, symbol: str, m5_by_symbol: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        m5 = m5_by_symbol.get(symbol)
        if m5 is None or m5.empty:
            return [{"symbol": symbol, "skipped": True, "reason": "no_m5"}]

        bar_time = m5.index[-1]
        bar_key = str(bar_time)
        if self._last_bar.get(symbol) == bar_key:
            return [{"symbol": symbol, "skipped": True, "reason": "same_bar"}]
        self._last_bar[symbol] = bar_key

        ctx = self._context(m5_by_symbol, bar_time)
        sched_ctx = SchedulerContext(ctx, self.scheduler.weight_allocator, self.scheduler.risk_manager)
        info = self.scheduler.state_machine.describe(sched_ctx, symbol)
        info["bar_time"] = bar_key
        info["close"] = float(m5["close"].iloc[-1])
        info["scheduler"] = True
        info["risk"] = self.scheduler.risk_manager.status()

        active = self.scheduler.state_machine.get_active_strategy()

        if active == "ai_model":
            predictions: dict[str, ModelPrediction] = {}
            ai = self.strategies["ai_model"]
            for sym in m5_by_symbol:
                sym_m5 = m5_by_symbol[sym]
                sym_ctx = self._context(m5_by_symbol, sym_m5.index[-1])
                sig = ai.on_bar(sym, sym_ctx)
                if sig is None:
                    continue
                predictions[sym] = self._strategy_signal_to_prediction(sig)

            outputs = self.scheduler.process_model_outputs(predictions, sched_ctx, primary_symbol=symbol)
            if not outputs:
                reject = "no_model_predictions"
                if predictions:
                    parts = [
                        f"{sym}:{pred.reject_reason or 'hold'}"
                        for sym, pred in predictions.items()
                    ]
                    reject = "; ".join(parts) if parts else "all_models_hold"
                flat = SchedulerOutput(
                    symbol=symbol,
                    direction="flat",
                    confidence=max((p.confidence for p in predictions.values()), default=0.0),
                    entry=float(m5["close"].iloc[-1]),
                    sl=0.0,
                    tp=0.0,
                    signal_id="",
                    market_state=info.get("state", ""),
                    reject_reason=reject,
                    metadata=sched_ctx.extra(),
                )
                return [self._scheduler_output_to_result(flat, info, self.signal_expiry)]

            results = [self._scheduler_output_to_result(o, info, self.signal_expiry) for o in outputs]
            self._save_state()
            return results

        strategy = self.strategies.get(active)
        if strategy is None:
            return [{**info, "error": f"unknown_strategy:{active}"}]
        sig = strategy.on_bar(symbol, ctx)
        return [self._strategy_result(sig, info, self.signal_expiry)]

    def tick_symbols(
        self,
        m5_by_symbol: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        syms = symbols or list(m5_by_symbol.keys())
        primary = self.primary_symbol
        if primary in syms:
            try:
                return self.on_bar(primary, m5_by_symbol)
            except Exception as ex:
                logger.exception("调度 tick 失败 %s: %s", primary, ex)
                return [{"symbol": primary, "error": str(ex), "scheduler": True}]
        return []

    def record_closed_trade(self, symbol: str, pnl_r: float) -> None:
        self.scheduler.record_trade_result(symbol, pnl_r, pnl_r > 0)
        self._save_state()


class SchedulerMt5Runner:
    """MT5 + 调度引擎 + 绘图管道。"""

    def __init__(self, config_path: str | Path, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent.parent
        p = Path(config_path)
        if not p.is_absolute():
            p = self.root / p
        if "config_scheduler" in p.name:
            self.config = load_scheduler_config(p)
            self.config["scheduler_enabled"] = True
            if "scheduler_core" not in self.config and self.config.get("weight_allocator"):
                self.config["scheduler_core"] = {
                    "weight_allocator": self.config.get("weight_allocator"),
                    "state_machine": self.config.get("state_machine"),
                    "risk_manager": self.config.get("risk_manager"),
                    "vote_margin": self.config.get("vote_margin", 0.1),
                    "min_emit_weight": self.config.get("min_emit_weight", 0.15),
                    "other_strategies": self.config.get("other_strategies"),
                }
        else:
            from zhulong.engine.multi_strategy_engine import load_multi_strategy_config

            base = load_multi_strategy_config(p)
            self.config = merge_scheduler_config(base, "config/config_scheduler.json")
        self.engine = SchedulerEngine(self.config, self.root)

        pipes = self.config.get("pipes") or {}
        from scripts.mt5_bridge import Mt5MarketData, PipeBridge

        self._bridge = PipeBridge(
            data_pipe=pipes.get("data_pipe", r"\\.\pipe\ZhuLong_Data"),
            drawing_pipe=pipes.get("drawing_pipe", r"\\.\pipe\ZhuLong_Drawing"),
        )
        self._mt5: dict[str, Mt5MarketData] = {}
        for sym, sc in (self.config.get("symbols") or {}).items():
            if not sc.get("enabled", True):
                continue
            bars = int(sc.get("m5_bars", self.config.get("m5_bars", 2000)))
            md = Mt5MarketData(sc.get("training_symbol", sym), broker_symbol=sc.get("broker_symbol"))
            md._bars = bars  # type: ignore[attr-defined]
            self._mt5[sym] = md

    def start(self) -> None:
        self._bridge.start()
        for sym, md in self._mt5.items():
            if not md.connect():
                logger.warning("[%s] MT5 未连接", sym)

    def stop(self) -> None:
        self._bridge.stop()
        for md in self._mt5.values():
            md.shutdown()

    def tick_once(self) -> list[dict]:
        m5_map = {}
        for sym, md in self._mt5.items():
            try:
                m5_map[sym] = md.fetch_m5(getattr(md, "_bars", 2000))
            except Exception as ex:
                logger.warning("[%s] M5 失败: %s", sym, ex)

        symbols = [s for s, sc in (self.config.get("symbols") or {}).items() if sc.get("enabled", True)]
        results = self.engine.tick_symbols(m5_map, symbols)
        for r in results:
            payload = r.get("draw_payload")
            if not payload:
                continue
            if self._bridge.send_draw(payload):
                logger.info(
                    "调度信号 [%s] %s %s conf=%.2f w=%s",
                    r.get("strategy"),
                    payload.get("signal_id"),
                    payload.get("direction"),
                    payload.get("confidence", 0),
                    (payload.get("meta") or {}).get("risk_weight"),
                )
                r["draw_sent"] = True
            else:
                r["draw_sent"] = False
        return results
