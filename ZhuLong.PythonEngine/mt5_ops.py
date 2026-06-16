"""MT5 操作 — 供 C# Python.NET 调用。"""

from __future__ import annotations

import time

# 与 ZhuLongIndicator.mq5 ServerOffsetSec() 对齐，避免重复探测 tick
_cached_server_offset_sec: int | None = None


def modify_sl_tp(ticket: int, sl: float, tp: float, deviation: int = 20) -> bool:
    import MetaTrader5 as mt5

    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p = pos[0]
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": p.symbol,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
    }
    result = mt5.order_send(req)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def close_partial(ticket: int, volume: float, deviation: int = 20) -> bool:
    import MetaTrader5 as mt5

    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p = pos[0]
    order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(p.symbol)
    if not tick:
        return False
    price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": p.symbol,
        "volume": volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": deviation,
    }
    result = mt5.order_send(req)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def close_full(ticket: int, deviation: int = 20) -> bool:
    import MetaTrader5 as mt5

    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    return close_partial(ticket, pos[0].volume, deviation)


def symbol_point(symbol: str) -> float:
    import MetaTrader5 as mt5

    info = mt5.symbol_info(symbol)
    return float(info.point) if info else 0.01


def offset_from_tick_delta(delta_sec: int) -> int:
    """由 tick.time - wall_utc 估算经纪商相对 UTC 的偏移（秒）。"""
    if abs(delta_sec) <= 120:
        return 0
    if abs(delta_sec) <= 14 * 3600:
        return int(round(delta_sec / 3600) * 3600)
    return 0


def _server_utc_offset_sec(*, refresh: bool = False) -> int:
    """与 MQL TimeTradeServer()-TimeGMT() 一致：copy_rates.time 须减掉此值才得真 UTC。

    部分终端（如 WCG build 5836）terminal_info 无 gmt_offset 字段，此时用活跃 tick 估算。
    勿用 terminal_info.timezone — 那是本地 PC 时区分钟数，不是经纪商服务器偏移。
    """
    global _cached_server_offset_sec
    if not refresh and _cached_server_offset_sec is not None:
        return _cached_server_offset_sec

    import MetaTrader5 as mt5

    ti = mt5.terminal_info()
    if ti is not None:
        gmt_offset = int(getattr(ti, "gmt_offset", 0) or 0)
        if gmt_offset:
            _cached_server_offset_sec = gmt_offset
            return gmt_offset

    utc_now = int(time.time())
    for sym in ("EURUSD", "XAUUSD", "USOIL", "BTCUSD"):
        if not mt5.symbol_select(sym, True):
            continue
        tick = mt5.symbol_info_tick(sym)
        if tick is None or int(tick.time) <= 0:
            continue
        off = offset_from_tick_delta(int(tick.time) - utc_now)
        if off or abs(int(tick.time) - utc_now) <= 120:
            _cached_server_offset_sec = off
            return off

    _cached_server_offset_sec = 0
    return 0


def _rate_time_to_utc_unix(unix_ts: int) -> int:
    return int(unix_ts) - _server_utc_offset_sec()


def fetch_m1_history(symbol: str, count: int = 1000):
    """拉取已收盘 M1（pos 从 1 起），供烛龙启动时预热 FeatureCache。"""
    import MetaTrader5 as mt5

    count = max(10, min(int(count), 5000))
    if not mt5.symbol_select(symbol, True):
        return []
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, count)
    if rates is None:
        return []

    out = []
    for r in rates:
        out.append(
            {
                "time": _rate_time_to_utc_unix(int(r["time"])),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["tick_volume"]),
            }
        )
    out.sort(key=lambda x: x["time"])
    return out
