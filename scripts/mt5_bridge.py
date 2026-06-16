#!/usr/bin/env python3
"""MT5 桥接：M5 K 线 + 命名管道服务（兼容 ZhuLongIndicator.mq5）。"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

try:
    import win32file
    import win32pipe
    import pywintypes
except ImportError:
    win32file = None  # type: ignore
    win32pipe = None  # type: ignore
    pywintypes = None  # type: ignore

import pandas as pd

BUF_SIZE = 65536
PIPE_UNLIMITED = 255


def normalize_pipe(name: str) -> str:
    if name.startswith(r"\\.\pipe\\"):
        return name.split(r"\\.\pipe\\")[-1]
    if name.startswith("\\\\.\\pipe\\"):
        return name.replace("\\\\.\\pipe\\", "")
    return name


OIL_BROKER_ALIASES = ("USOIL", "XTIUSD", "WTI", "CL-OIL", "USOILm", "XTIUSDm", "#USOIL")
XAU_BROKER_ALIASES = ("XAUUSD", "XAUUSDm", "GOLD", "#XAUUSD")


class Mt5MarketData:
    """从 MT5 拉取 M5/M1；支持 training_symbol → broker_symbol 映射。"""

    def __init__(self, symbol: str = "XAUUSD", broker_symbol: str | None = None) -> None:
        self.symbol = symbol
        self._requested_broker = broker_symbol
        self._broker_symbol: str | None = None

    def _alias_list(self) -> list[str]:
        base = [self._requested_broker or self.symbol, self.symbol]
        if self.symbol.upper() in ("USOIL", "WTI", "OIL") or (
            self._requested_broker and "OIL" in self._requested_broker.upper()
        ):
            base.extend(OIL_BROKER_ALIASES)
        else:
            base.extend(XAU_BROKER_ALIASES)
        seen: list[str] = []
        for s in base:
            if s and s not in seen:
                seen.append(s)
        return seen

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 未安装")
            return False
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.error("MT5 initialize 失败: %s", mt5.last_error())
                return False
        broker = None
        for sym in self._alias_list():
            if mt5.symbol_select(sym, True):
                broker = sym
                break
        if broker is None:
            logger.error("无法选择品种 %s（已尝试 %s）", self.symbol, self._alias_list())
            return False
        self._broker_symbol = broker
        if broker != (self._requested_broker or self.symbol):
            logger.info("符号映射 %s → %s", self._requested_broker or self.symbol, broker)
        return True

    def shutdown(self) -> None:
        if mt5 is not None:
            mt5.shutdown()

    @property
    def broker_symbol(self) -> str:
        return self._broker_symbol or self.symbol

    def fetch_m5(self, bars: int = 2000) -> pd.DataFrame:
        if mt5 is None or not self.connect():
            raise RuntimeError("MT5 未连接")
        rates = mt5.copy_rates_from_pos(self.broker_symbol, mt5.TIMEFRAME_M5, 0, bars)
        if rates is None or len(rates) < 60:
            raise RuntimeError(f"M5 不足: {0 if rates is None else len(rates)}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = df.get("real_volume", 1.0)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def last_m5_time(self) -> pd.Timestamp | None:
        df = self.fetch_m5(bars=2)
        return df.index[-1] if len(df) else None


class PipeBridge:
    """
    命名管道服务端（与 ZhuLong.exe PipeServer 协议一致）。
    - ZhuLong_Data: INBOUND，接收 MT5 M1 JSON
    - ZhuLong_Drawing: OUTBOUND，向 MT5 发送绘图 JSON
    """

    def __init__(
        self,
        data_pipe: str = r"\\.\pipe\ZhuLong_Data",
        drawing_pipe: str = r"\\.\pipe\ZhuLong_Drawing",
        on_bar: Callable[[dict], None] | None = None,
    ) -> None:
        self._data_name = normalize_pipe(data_pipe)
        self._draw_name = normalize_pipe(drawing_pipe)
        self._on_bar = on_bar
        self._stop = threading.Event()
        self._bar_queue: Queue = Queue()
        self._draw_handle = None
        self._draw_lock = threading.Lock()
        self._data_thread: threading.Thread | None = None
        self._draw_thread: threading.Thread | None = None
        self.data_connected = False
        self.draw_connected = False

    def start(self) -> None:
        if win32pipe is None:
            raise RuntimeError("需要 pywin32: pip install pywin32")
        self._data_thread = threading.Thread(target=self._data_loop, name="PipeData", daemon=True)
        self._draw_thread = threading.Thread(target=self._draw_loop, name="PipeDraw", daemon=True)
        self._data_thread.start()
        self._draw_thread.start()
        logger.info("管道服务已启动 %s / %s", self._data_name, self._draw_name)

    def stop(self) -> None:
        self._stop.set()

    def poll_bar(self, timeout: float = 0.0) -> dict | None:
        try:
            return self._bar_queue.get(timeout=timeout)
        except Empty:
            return None

    def send_draw(self, payload: dict) -> bool:
        if win32file is None:
            return False
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._draw_lock:
            handle = self._draw_handle
            if handle is None:
                logger.warning("绘图管道未连接，跳过 draw_signal")
                return False
            try:
                win32file.WriteFile(handle, line)
                return True
            except pywintypes.error as exc:
                logger.warning("绘图发送失败: %s", exc)
                self.draw_connected = False
                return False

    def _data_loop(self) -> None:
        while not self._stop.is_set():
            handle = None
            try:
                handle = win32pipe.CreateNamedPipe(
                    f"\\\\.\\pipe\\{self._data_name}",
                    win32pipe.PIPE_ACCESS_INBOUND,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                    PIPE_UNLIMITED,
                    BUF_SIZE,
                    BUF_SIZE,
                    0,
                    None,
                )
                logger.info("等待 MT5 连接 %s", self._data_name)
                win32pipe.ConnectNamedPipe(handle, None)
                self.data_connected = True
                logger.info("MT5 已连接 %s", self._data_name)
                self._read_data(handle)
            except pywintypes.error as exc:
                if not self._stop.is_set():
                    logger.warning("数据管道异常: %s", exc)
                    time.sleep(3)
            finally:
                self.data_connected = False
                if handle is not None:
                    try:
                        win32file.CloseHandle(handle)
                    except pywintypes.error:
                        pass

    def _read_data(self, handle) -> None:
        buffer = b""
        while not self._stop.is_set():
            try:
                _, chunk = win32file.ReadFile(handle, BUF_SIZE)
            except pywintypes.error:
                break
            if not chunk:
                time.sleep(0.05)
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "bar":
                        self._bar_queue.put(msg)
                        if self._on_bar:
                            self._on_bar(msg)
                except json.JSONDecodeError:
                    pass

    def _draw_loop(self) -> None:
        while not self._stop.is_set():
            handle = None
            try:
                handle = win32pipe.CreateNamedPipe(
                    f"\\\\.\\pipe\\{self._draw_name}",
                    win32pipe.PIPE_ACCESS_OUTBOUND,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                    PIPE_UNLIMITED,
                    BUF_SIZE,
                    BUF_SIZE,
                    0,
                    None,
                )
                logger.info("等待 MT5 连接 %s", self._draw_name)
                win32pipe.ConnectNamedPipe(handle, None)
                with self._draw_lock:
                    self._draw_handle = handle
                self.draw_connected = True
                logger.info("MT5 已连接 %s", self._draw_name)
                while not self._stop.is_set() and self.draw_connected:
                    time.sleep(0.5)
            except pywintypes.error as exc:
                if not self._stop.is_set():
                    logger.warning("绘图管道异常: %s", exc)
                    time.sleep(3)
            finally:
                with self._draw_lock:
                    self._draw_handle = None
                self.draw_connected = False
                if handle is not None:
                    try:
                        win32file.CloseHandle(handle)
                    except pywintypes.error:
                        pass


def smoke_draw(symbol: str = "XAUUSD") -> int:
    """需 PipeBridge 或 ZhuLong.exe 正在运行。"""
    bridge = PipeBridge()
    bridge.start()
    time.sleep(2)
    payload = {
        "action": "draw_signal",
        "signal_id": f"SMOKE_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
        "symbol": symbol,
        "direction": "buy",
        "entry": 2350.0,
        "sl": 2340.0,
        "tp": 2370.0,
        "confidence": 0.88,
        "expiry_minutes": 240,
    }
    ok = bridge.send_draw(payload)
    bridge.stop()
    print("draw ok=" if ok else "draw failed", payload)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    md = Mt5MarketData("XAUUSD")
    if md.connect():
        t = md.last_m5_time()
        print("MT5 OK, last M5:", t)
        md.shutdown()
    else:
        print("MT5 fail")
        sys.exit(1)
