"""多策略总控：状态机 + 策略池 + 信号输出。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from zhulong.strategies.ai_model import AIModelStrategy
from zhulong.strategies.base import StrategyContext, StrategySignal
from zhulong.strategies.grid_system import GridSystem
from zhulong.strategies.spread_hedge import SpreadHedge
from zhulong.strategies.state_machine import MarketState, StrategyStateMachine
from zhulong.strategies.trend_system import TrendSystem

logger = logging.getLogger(__name__)


def load_multi_strategy_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    text = p.read_text(encoding="utf-8-sig")
    return json.loads(text)


class MultiStrategyEngine:
    def __init__(self, config: dict[str, Any], root: Path | None = None) -> None:
        self.config = config
        self.root = root or Path.cwd()
        self.enabled = bool(config.get("enabled", True))
        self.signal_expiry = int(config.get("signal_expiry_minutes", 240))
        self.macro_silence = bool(config.get("macro_silence", False))

        self.state_machine = StrategyStateMachine(config)
        self.strategies = self._build_strategies()
        self._last_bar: dict[str, str] = {}

    @property
    def primary_symbol(self) -> str:
        return self.state_machine.primary_symbol

    def set_primary_symbol(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.state_machine.primary_symbol = sym

    def _build_strategies(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "ai_model": AIModelStrategy({"symbols": cfg.get("symbols") or {}}, self.root),
            "trend_system": TrendSystem(cfg.get("trend_system") or {}),
            "spread_hedge": SpreadHedge(cfg.get("spread_hedge") or {}),
            "grid_system": GridSystem(cfg.get("grid_system") or {}),
        }

    def _context(self, m5_by_symbol: dict[str, pd.DataFrame], bar_time: pd.Timestamp | None) -> StrategyContext:
        return StrategyContext(
            m5_by_symbol=m5_by_symbol,
            config=self.config,
            macro_silence=self.macro_silence,
            bar_time=bar_time,
        )

    def on_bar(
        self,
        symbol: str,
        m5_by_symbol: dict[str, pd.DataFrame],
        *,
        force_strategy: str | None = None,
    ) -> dict[str, Any]:
        """处理单品种新 M5 bar，返回调度结果。"""
        m5 = m5_by_symbol.get(symbol)
        if m5 is None or m5.empty:
            return {"symbol": symbol, "skipped": True, "reason": "no_m5"}

        bar_time = m5.index[-1]
        bar_key = str(bar_time)
        if self._last_bar.get(symbol) == bar_key:
            return {"symbol": symbol, "skipped": True, "reason": "same_bar"}
        self._last_bar[symbol] = bar_key

        ctx = self._context(m5_by_symbol, bar_time)
        state = self.state_machine.get_current_state(ctx, symbol)
        active_name = force_strategy or self.state_machine.select_strategy(state)
        strategy = self.strategies.get(active_name)
        if strategy is None:
            return {
                "symbol": symbol,
                "state": state.value,
                "strategy": active_name,
                "error": "unknown_strategy",
            }

        sig = strategy.on_bar(symbol, ctx)
        info = self.state_machine.describe(ctx, symbol)
        info["bar_time"] = bar_key
        info["close"] = float(m5["close"].iloc[-1])

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
            result["draw_payload"] = sig.to_draw_payload(self.signal_expiry)
        return result

    def tick_symbols(
        self,
        m5_by_symbol: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        syms = symbols or list(m5_by_symbol.keys())
        primary = self.state_machine.primary_symbol
        ordered = [primary] + [s for s in syms if s != primary]
        out: list[dict] = []
        for sym in ordered:
            if sym not in m5_by_symbol:
                continue
            try:
                out.append(self.on_bar(sym, m5_by_symbol))
            except Exception as ex:
                logger.exception("多策略 tick 失败 %s: %s", sym, ex)
                out.append({"symbol": sym, "error": str(ex)})
        return out


class MultiStrategyMt5Runner:
    """MT5 拉 M5 + 多策略引擎 + 绘图管道。"""

    def __init__(self, config_path: str | Path, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent
        self.config = load_multi_strategy_config(config_path)
        self.engine = MultiStrategyEngine(self.config, self.root)
        pipes = self.config.get("pipes") or {}
        from scripts.mt5_bridge import Mt5MarketData, PipeBridge

        self._bridge = PipeBridge(
            data_pipe=pipes.get("data_pipe", r"\\.\pipe\ZhuLong_Data"),
            drawing_pipe=pipes.get("drawing_pipe", r"\\.\pipe\ZhuLong_Drawing"),
        )
        self._mt5: dict[str, Mt5MarketData] = {}
        sym_cfg = self.config.get("symbols") or {}
        for sym, sc in sym_cfg.items():
            if not sc.get("enabled", True):
                continue
            bars = int(sc.get("m5_bars", self.config.get("m5_bars", 2000)))
            md = Mt5MarketData(sc.get("training_symbol", sym), broker_symbol=sc.get("broker_symbol"))
            md._bars = bars  # type: ignore[attr-defined]
            self._mt5[sym] = md

    def _fetch_all_m5(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym, md in self._mt5.items():
            bars = getattr(md, "_bars", 2000)
            try:
                out[sym] = md.fetch_m5(bars)
            except Exception as ex:
                logger.warning("[%s] M5 失败: %s", sym, ex)
        return out

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
        m5_map = self._fetch_all_m5()
        symbols = [s for s, sc in (self.config.get("symbols") or {}).items() if sc.get("enabled", True)]
        results = self.engine.tick_symbols(m5_map, symbols)
        for r in results:
            payload = r.get("draw_payload")
            if not payload:
                continue
            if self._bridge.send_draw(payload):
                logger.info(
                    "信号已发送 [%s] %s %s conf=%.2f",
                    r.get("strategy"),
                    payload.get("signal_id"),
                    payload.get("direction"),
                    payload.get("confidence", 0),
                )
                r["draw_sent"] = True
            else:
                logger.warning("绘图管道未就绪: %s", payload.get("signal_id"))
                r["draw_sent"] = False
        return results
