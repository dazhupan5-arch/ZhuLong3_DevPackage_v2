"""持仓扫描与管理（G5/G8/G10）。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from zhulong.config_loader import Config
from zhulong.db.repository import get_connection
from zhulong.mt5_bridge import Mt5Bridge
from zhulong.utils.paths import broker_symbol

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore


@dataclass
class ManagedPosition:
    ticket: int
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    open_time: int
    volume: float
    peak_profit_pct: float = 0.0
    partial_step: int = 0
    trailing_sl: Optional[float] = None


class PositionManagerThread(threading.Thread):
    def __init__(
        self,
        config: Config,
        bridge: Mt5Bridge,
        pending_signals: list,
        pending_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="PositionManagerThread", daemon=True)
        self._config = config
        self._bridge = bridge
        self._pending = pending_signals
        self._lock = pending_lock
        self._stop = stop_event
        self._managed: dict[int, ManagedPosition] = {}

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._bridge.connected:
                    self._scan()
            except Exception as exc:
                logger.exception("持仓扫描异常: %s", exc)
            self._stop.wait(1.0)

    def _scan(self) -> None:
        self._match_new()
        self._update_managed()

    def _match_new(self) -> None:
        pm = self._config.get("position_management", default={}) or {}
        tol = pm.get("match_price_tolerance_points", 5)
        window = pm.get("match_time_window_seconds", 60)
        mapping = self._config.get("symbol_mapping", default={}) or {}
        prefix = self._config.get("mt5", "comment_prefix", default="ZhuLong")

        with self._lock:
            # ===== P2-1: 单信号约束 — 过滤掉已有活跃持仓品种的 pending 信号 =====
            active_symbols = {mp.symbol for mp in self._managed.values()}
            pending = [s for s in list(self._pending) if s.symbol not in active_symbols]
            # ===== 结束 =====

        positions = self._bridge.positions()
        now = time.time()
        for sig in pending:
            broker_sym = broker_symbol(sig.symbol, mapping)
            for p in positions:
                if p.ticket in self._managed:
                    continue
                comment = getattr(p, "comment", "") or ""
                matched = False
                if comment.startswith(f"{prefix}_"):
                    matched = sig.signal_id in comment
                else:
                    point = self._bridge.symbol_point(sig.symbol)
                    price_ok = abs(p.price_open - sig.entry_price) <= tol * point
                    time_ok = abs(p.time - sig.created_at) <= window
                    dir_ok = (
                        (sig.direction == "buy" and p.type == mt5.POSITION_TYPE_BUY)
                        or (sig.direction == "sell" and p.type == mt5.POSITION_TYPE_SELL)
                    )
                    sym_ok = p.symbol == broker_sym
                    matched = price_ok and time_ok and dir_ok and sym_ok
                    if matched:
                        logger.warning("持仓匹配降级（无 Comment）ticket=%s", p.ticket)

                if not matched:
                    continue
                self._managed[p.ticket] = ManagedPosition(
                    ticket=p.ticket,
                    signal_id=sig.signal_id,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    entry_price=float(p.price_open),
                    open_time=int(p.time),
                    volume=float(p.volume),
                )
                logger.info("已匹配托管持仓 ticket=%s signal=%s", p.ticket, sig.signal_id)
                break

    def _profit_pct(self, pos, mp: ManagedPosition) -> float:
        tick = mt5.symbol_info_tick(pos.symbol) if mt5 else None
        if not tick:
            return 0.0
        price = tick.bid if mp.direction == "buy" else tick.ask
        if mp.direction == "buy":
            return (price - mp.entry_price) / mp.entry_price * 100
        return (mp.entry_price - price) / mp.entry_price * 100

    def _update_managed(self) -> None:
        pm = self._config.get("position_management", default={}) or {}
        max_hold = pm.get("max_hold_minutes", 240) * 60
        retry = pm.get("order_retry_max", 3)

        for ticket, mp in list(self._managed.items()):
            pos_list = mt5.positions_get(ticket=ticket) if mt5 else None
            if not pos_list:
                del self._managed[ticket]
                continue
            pos = pos_list[0]
            profit = self._profit_pct(pos, mp)
            mp.peak_profit_pct = max(mp.peak_profit_pct, profit)

            # ===== P0-1: 时间停止改为"到期且未盈利才平仓" =====
            if time.time() - mp.open_time >= max_hold and profit <= 0:
                self._close(ticket, mp, "time_stop", retry)
                continue
            # ===== 结束 =====

            dd_ratio = pm.get("max_drawdown_ratio", 0.4)
            if mp.peak_profit_pct > 0 and profit < mp.peak_profit_pct * (1 - dd_ratio):
                self._close(ticket, mp, "trailing", retry)
                continue

            act = pm.get("trailing_activation_pct", 0.15)
            step = pm.get("trailing_step_pct", 0.10)
            if profit >= act:
                self._apply_trailing(pos, mp, step, retry)

            self._partial_tp(pos, mp, pm, retry)

    def _apply_trailing(self, pos, mp: ManagedPosition, step_pct: float, retry: int) -> None:
        step = mp.entry_price * step_pct / 100.0
        if mp.direction == "buy":
            new_sl = max(pos.sl, mp.entry_price) if pos.sl else mp.entry_price
            if pos.sl and pos.sl < mp.entry_price + step:
                new_sl = mp.entry_price + step
        else:
            new_sl = min(pos.sl, mp.entry_price) if pos.sl else mp.entry_price
            if pos.sl and pos.sl > mp.entry_price - step:
                new_sl = mp.entry_price - step
        if mp.trailing_sl != new_sl:
            for _ in range(retry):
                if self._bridge.modify_sl_tp(ticket=pos.ticket, sl=new_sl, tp=pos.tp):
                    mp.trailing_sl = new_sl
                    break

    def _partial_tp(self, pos, mp: ManagedPosition, pm: dict, retry: int) -> None:
        profit = self._profit_pct(pos, mp)
        t1 = pm.get("partial_target1_pct", 0.25)
        r1 = pm.get("partial_ratio1", 0.5)
        t2 = pm.get("partial_target2_pct", 0.40)
        r2 = pm.get("partial_ratio2", 0.5)
        if mp.partial_step == 0 and profit >= t1:
            vol = round(pos.volume * r1, 2)
            if vol > 0 and self._bridge.close_partial(pos.ticket, vol):
                mp.partial_step = 1
        elif mp.partial_step == 1 and profit >= t2:
            vol = round(pos.volume * r2, 2)
            if vol > 0 and self._bridge.close_partial(pos.ticket, vol):
                mp.partial_step = 2

    def _close(self, ticket: int, mp: ManagedPosition, reason: str, retry: int) -> None:
        for _ in range(retry):
            if self._bridge.close_full(ticket):
                logger.info("平仓 ticket=%s reason=%s", ticket, reason)
                self._managed.pop(ticket, None)
                return
