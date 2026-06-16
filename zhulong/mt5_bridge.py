"""MT5 Python API 桥接。"""

from __future__ import annotations

import logging
from typing import Any

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from zhulong.config_loader import Config
from zhulong.utils.paths import broker_symbol

logger = logging.getLogger(__name__)


class Mt5Bridge:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and mt5 is not None and mt5.terminal_info() is not None

    def initialize(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 包未安装")
            return False
        if self._connected and self.connected:
            return True
        if not mt5.initialize():
            logger.error("mt5.initialize 失败: %s", mt5.last_error())
            self._connected = False
            return False
        self._connected = True
        info = mt5.terminal_info()
        logger.info("MT5 已连接: %s", info.name if info else "unknown")
        return True

    def shutdown(self) -> None:
        if mt5 and self._connected:
            mt5.shutdown()
        self._connected = False

    def ensure_connection(self) -> bool:
        if self.connected:
            return True
        return self.initialize()

    def resolve_symbol(self, standard: str) -> str:
        mapping = self._config.get("symbol_mapping", default={}) or {}
        return broker_symbol(standard, mapping)

    def symbol_point(self, standard: str) -> float:
        sym = self.resolve_symbol(standard)
        if not self.ensure_connection():
            return 0.01
        info = mt5.symbol_info(sym)
        return float(info.point) if info else 0.01

    def positions(self) -> list[Any]:
        if not self.ensure_connection():
            return []
        items = mt5.positions_get()
        return list(items) if items else []

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        if not self.ensure_connection():
            return False
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        p = pos[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": p.symbol,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            logger.warning("改 SL/TP 失败 ticket=%s: %s", ticket, result)
        return ok

    def close_partial(self, ticket: int, volume: float) -> bool:
        if not self.ensure_connection():
            return False
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        p = pos[0]
        order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            return False
        price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self._config.get("mt5", "deviation", default=20),
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            logger.warning("部分平仓失败 ticket=%s: %s", ticket, result)
        return ok

    def close_full(self, ticket: int) -> bool:
        if not self.ensure_connection():
            return False
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        return self.close_partial(ticket, pos[0].volume)
