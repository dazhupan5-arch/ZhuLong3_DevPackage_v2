"""
命名管道数据接收（G4 / G10）。
Python 作为管道服务端，阻塞等待 MT5 指标连接并推送 M1 K 线 JSON。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from queue import Queue
from typing import Callable, Optional

try:
    import win32file
    import win32pipe
    import pywintypes
except ImportError:
    win32file = None  # type: ignore
    win32pipe = None  # type: ignore
    pywintypes = None  # type: ignore

logger = logging.getLogger(__name__)

BUF_SIZE = 65536


class DataReceiverThread(threading.Thread):
    """读取 M1 K 线并回调 on_bar。"""

    def __init__(
        self,
        pipe_name: str,
        bar_queue: Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="DataReceiverThread", daemon=True)
        self._pipe_name = pipe_name
        self._bar_queue = bar_queue
        self._stop = stop_event

    def run(self) -> None:
        if win32pipe is None:
            logger.error("pywin32 未安装，无法创建命名管道")
            return

        while not self._stop.is_set():
            handle = None
            try:
                handle = win32pipe.CreateNamedPipe(
                    self._pipe_name,
                    win32pipe.PIPE_ACCESS_INBOUND,
                    win32pipe.PIPE_TYPE_MESSAGE
                    | win32pipe.PIPE_READMODE_MESSAGE
                    | win32pipe.PIPE_WAIT,
                    1,
                    BUF_SIZE,
                    BUF_SIZE,
                    0,
                    None,
                )
                logger.info("等待 MT5 连接数据管道: %s", self._pipe_name)
                win32pipe.ConnectNamedPipe(handle, None)
                logger.info("MT5 已连接数据管道")
                self._read_loop(handle)
            except pywintypes.error as exc:
                if not self._stop.is_set():
                    logger.warning("数据管道异常，3s 后重建: %s", exc)
                    time.sleep(3)
            finally:
                if handle is not None:
                    try:
                        win32file.CloseHandle(handle)
                    except pywintypes.error:
                        pass

    def _read_loop(self, handle) -> None:
        buffer = ""
        while not self._stop.is_set():
            try:
                _, data = win32file.ReadFile(handle, BUF_SIZE)
            except pywintypes.error:
                break
            if not data:
                continue
            buffer += data.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "bar":
                        self._bar_queue.put(msg)
                except json.JSONDecodeError:
                    logger.debug("忽略非法 JSON: %s", line[:120])
