"""
下行绘图管道客户端：向 MT5 指标发送 draw_signal / clear_signal JSON。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

try:
    import win32file
    import win32pipe
    import pywintypes
except ImportError:
    win32file = None  # type: ignore
    win32pipe = None  # type: ignore
    pywintypes = None  # type: ignore

logger = logging.getLogger(__name__)


class DrawingClient:
    def __init__(self, pipe_name: str) -> None:
        self._pipe_name = pipe_name
        self._lock = threading.Lock()

    def send(self, payload: dict) -> bool:
        if win32pipe is None:
            logger.error("pywin32 未安装，无法发送绘图指令")
            return False
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")
        with self._lock:
            for attempt in range(3):
                handle = None
                try:
                    handle = win32file.CreateFile(
                        self._pipe_name,
                        win32file.GENERIC_WRITE,
                        0,
                        None,
                        win32file.OPEN_EXISTING,
                        0,
                        None,
                    )
                    win32file.WriteFile(handle, data)
                    return True
                except pywintypes.error as exc:
                    logger.debug("绘图管道发送失败 (%s/3): %s", attempt + 1, exc)
                    time.sleep(0.5)
                finally:
                    if handle is not None:
                        try:
                            win32file.CloseHandle(handle)
                        except pywintypes.error:
                            pass
        return False

    def draw_signal(self, signal: dict) -> bool:
        payload = {
            "action": "draw_signal",
            "signal_id": signal["signal_id"],
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "entry": signal["entry_price"],
            "sl": signal["stop_loss"],
            "tp": signal["take_profit"],
            "confidence": signal["confidence"],
            "expiry_minutes": signal.get("expiry_minutes", 240),
        }
        if "strategy" in signal:
            payload["strategy"] = signal["strategy"]
        return self.send(payload)

    def clear_signal(self, signal_id: str) -> bool:
        return self.send({"action": "clear_signal", "signal_id": signal_id})
