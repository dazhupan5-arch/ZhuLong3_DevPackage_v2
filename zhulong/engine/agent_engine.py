"""AgentEngine：配置加载与 tick 入口。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from zhulong.engine.runtime_config import apply_runtime_primary, bind_engine_primary
from zhulong.utils.json_safe import json_safe
from zhulong.utils.paths import resolve_agent_config_path

logger = logging.getLogger(__name__)

_engine_cache: dict[str, "AgentEngine"] = {}


def get_or_create_engine(req: dict[str, Any], root: Path) -> tuple["AgentEngine", str | None]:
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = resolve_agent_config_path(str(cfg_rel), root)
    config = load_agent_config(cfg_path, root=root)
    merged, runtime_primary = merge_agent_runtime(config, req)
    key = str(cfg_path.resolve())
    engine = _engine_cache.get(key)
    if engine is None:
        engine = AgentEngine(merged, root=root)
        _engine_cache[key] = engine
    elif runtime_primary:
        bind_engine_primary(engine, runtime_primary)
    return engine, runtime_primary


def warm_engine_cache(req: dict[str, Any], root: Path) -> AgentEngine:
    """预创建 AgentEngine 并热起全栈（Horizon/KN2/RL），避免首 tick 冷加载。"""
    engine, _ = get_or_create_engine(req, root)
    agent = engine.agent
    agent._ensure_kn2()
    agent._ensure_rl()
    return engine


def load_agent_config(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (root or Path.cwd()) / p
    if not p.is_file():
        return {"enabled": False}
    return json.loads(p.read_text(encoding="utf-8-sig"))


class AgentEngine:
    def __init__(self, config: dict[str, Any], root: Path | None = None) -> None:
        from zhulong.agent.trading_agent import TradingAgent

        self.config = config
        self.root = root or Path.cwd()
        self.agent = TradingAgent(config, root=self.root)
        self.primary_symbol = self.agent.primary_symbol

    def set_primary_symbol(self, symbol: str) -> None:
        self.agent.set_primary_symbol(symbol)
        self.primary_symbol = self.agent.primary_symbol

    def tick_symbols(
        self,
        m5_by_symbol: dict[str, pd.DataFrame],
        symbols: list[str] | None = None,
        account: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.agent.tick_symbols(m5_by_symbol, symbols, account)

    def record_closed_trade(self, symbol: str, pnl_r: float) -> None:
        self.agent.record_closed_trade(symbol, pnl_r)

    def record_signal_emitted(self, symbol: str) -> None:
        self.agent.record_signal_emitted(symbol)


def merge_agent_runtime(config: dict[str, Any], req: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    cfg = dict(config)
    runtime_primary = apply_runtime_primary(cfg, req.get("primary_symbol"))
    if runtime_primary:
        cfg["primary_symbol"] = runtime_primary
    if req.get("account"):
        cfg["_runtime_account"] = req["account"]
    return cfg, runtime_primary


def record_agent_closed_trade(
    symbol: str,
    pnl_r: float,
    req: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = resolve_agent_config_path(str(cfg_rel), root)
    config = load_agent_config(cfg_path, root=root)
    if not config.get("enabled", True):
        return {"ok": False, "error": "agent_disabled_in_config"}
    try:
        engine, runtime_primary = get_or_create_engine(req, root)
        if runtime_primary:
            bind_engine_primary(engine, runtime_primary)
        engine.record_closed_trade(symbol.strip().upper(), float(pnl_r))
        return {"ok": True, "agent": True, "symbol": symbol, "pnl_r": float(pnl_r)}
    except Exception as ex:
        logger.exception("record_closed_trade 失败")
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


def record_agent_signal_emitted(
    symbol: str,
    req: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = resolve_agent_config_path(str(cfg_rel), root)
    config = load_agent_config(cfg_path, root=root)
    if not config.get("enabled", True):
        return {"ok": False, "error": "agent_disabled_in_config"}
    try:
        engine, runtime_primary = get_or_create_engine(req, root)
        if runtime_primary:
            bind_engine_primary(engine, runtime_primary)
        engine.record_signal_emitted(symbol.strip().upper())
        return {"ok": True, "agent": True, "symbol": symbol}
    except Exception as ex:
        logger.exception("record_signal_emitted 失败")
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


def run_agent_tick(
    m5_by_symbol: dict[str, pd.DataFrame],
    req: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    cfg_rel = req.get("config_path") or "config/config_agent.json"
    cfg_path = resolve_agent_config_path(str(cfg_rel), root)
    config = load_agent_config(cfg_path, root=root)
    if not config.get("enabled", True):
        reason = "agent_disabled"
        if not cfg_path.is_file():
            reason = f"config_not_found:{cfg_path}"
        elif config.get("enabled") is False:
            reason = "agent_disabled_in_config"
        return {"ok": True, "agent": False, "results": [], "reason": reason, "config_path": str(cfg_path)}

    if bool(req.get("macro_silence")):
        symbols = req.get("symbols") or list(m5_by_symbol.keys())
        return {
            "ok": True,
            "agent": True,
            "results": [
                {"symbol": str(s), "skipped": True, "reason": "macro_silence", "strategy": "rl_agent"}
                for s in symbols
            ],
        }

    symbols = req.get("symbols") or list(m5_by_symbol.keys())
    try:
        engine, runtime_primary = get_or_create_engine(req, root)

        account = req.get("account") or {}
        ticks = req.get("ticks_by_symbol") or {}
        positions = req.get("open_positions") or []
        if ticks:
            account["_ticks"] = ticks
        if positions:
            account["_positions"] = positions
        if "m5_includes_forming" in req:
            account["_m5_includes_forming"] = bool(req.get("m5_includes_forming", True))
        if req.get("decision_bar_unix"):
            account["_decision_bar_unix"] = int(req["decision_bar_unix"])
        macro_feats = req.get("macro_features")
        if macro_feats:
            account["_macro_features"] = macro_feats
        results = engine.tick_symbols(m5_by_symbol, symbols, account)
        if not results:
            return {
                "ok": False,
                "agent": True,
                "error": f"agent_empty_results primary={engine.primary_symbol} symbols={symbols}",
            }
        safe_results = [json_safe(r) if isinstance(r, dict) else json_safe(r) for r in results]
        skipped_only = all(isinstance(r, dict) and r.get("skipped") for r in safe_results)
        return json_safe({
            "ok": True,
            "agent": True,
            "primary_symbol": runtime_primary or engine.primary_symbol,
            "results": safe_results,
            "skipped_only": skipped_only,
        })
    except Exception as ex:
        logger.exception("智能体 tick 失败")
        return {"ok": False, "agent": False, "error": f"{type(ex).__name__}: {ex}"}


class AgentMt5Runner:
    """MT5 + TradingAgent + 绘图管道。"""

    def __init__(self, config_path: str | Path, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent.parent
        p = Path(config_path)
        if not p.is_absolute():
            p = self.root / p
        self.config = load_agent_config(p)
        self.engine = AgentEngine(self.config, self.root)

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
                    "智能体信号 [%s] %s %s conf=%.2f action=%s",
                    r.get("strategy"),
                    payload.get("signal_id"),
                    payload.get("direction"),
                    payload.get("confidence", 0),
                    r.get("action"),
                )
                r["draw_sent"] = True
            else:
                r["draw_sent"] = False
        return results

