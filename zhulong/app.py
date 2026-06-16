"""应用编排：五线程模型（G10）。"""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from zhulong.config_loader import Config
from zhulong.data_receiver import DataReceiverThread
from zhulong.drawing_client import DrawingClient
from zhulong.feature_engine import (
    BarStore,
    compute_hourly_background,
    compute_m5_features,
    latest_sequence,
)
from zhulong.inference_engine import InferenceEngine
from zhulong.macro_calendar import MacroCalendarThread, macro_features
from zhulong.mt5_bridge import Mt5Bridge
from zhulong.position_manager import PositionManagerThread
from zhulong.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class SignalSchedulerThread(threading.Thread):
    """每 5 分钟对齐执行推理与信号过滤。"""

    def __init__(
        self,
        config: Config,
        bar_store: BarStore,
        inference: InferenceEngine,
        signal_gen: SignalGenerator,
        drawing: DrawingClient,
        pending_signals: list,
        pending_lock: threading.Lock,
        on_signal: Callable,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="SignalSchedulerThread", daemon=True)
        self._config = config
        self._store = bar_store
        self._inference = inference
        self._signal_gen = signal_gen
        self._drawing = drawing
        self._pending = pending_signals
        self._lock = pending_lock
        self._on_signal = on_signal
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            delay = 300 - (time.time() % 300)
            self._stop.wait(delay)
            if self._stop.is_set():
                break
            self._tick()

    def _tick(self) -> None:
        symbols = self._config.get("model", "default_symbols", default=["XAUUSD"])
        atr_cfg = self._config.get("atr_channel", default={}) or {}
        seq_len = int(self._config.get("model", "seq_len", default=60))

        for symbol in symbols:
            m5 = self._store.m5.get(symbol)
            if m5 is None or len(m5) < seq_len + 20:
                continue
            try:
                m5_feat = compute_m5_features(
                    m5,
                    atr_period=atr_cfg.get("period", 14),
                    ema_fast=atr_cfg.get("ema_fast", 30),
                    ema_slow=atr_cfg.get("ema_slow", 60),
                )
                seq = latest_sequence(m5_feat, seq_len)
                if seq is None:
                    continue
                hourly = compute_hourly_background(m5)
                macro = macro_features()
                pred = self._inference.predict(symbol, seq, hourly, macro)
                sig, reason = self._signal_gen.try_generate(symbol, m5, m5_feat, seq, hourly, pred)
                if sig is None:
                    logger.debug("无信号 %s: %s", symbol, reason)
                    continue
                with self._lock:
                    self._pending.append(sig)
                self._drawing.draw_signal(sig.__dict__)
                self._on_signal(sig, reason)
                logger.info("新信号 %s %s conf=%.2f", sig.symbol, sig.direction, sig.confidence)
            except Exception as exc:
                logger.exception("信号调度失败 %s: %s", symbol, exc)


class ZhuLongApp:
    def __init__(self, config: Config, on_log: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        self._on_log = on_log or (lambda m: None)
        self._stop = threading.Event()
        self._bar_queue: Queue = Queue()
        self._bar_store = BarStore()
        self._pending: list = []
        self._pending_lock = threading.Lock()
        self._bridge = Mt5Bridge(config)
        self._drawing = DrawingClient(config.get("pipes", "drawing_pipe"))
        self._inference = InferenceEngine(config.get("model", default={}) or {})
        self._signal_gen = SignalGenerator(config)
        self._threads: list[threading.Thread] = []
        self._bar_thread: Optional[threading.Thread] = None

    def log(self, msg: str) -> None:
        logger.info(msg)
        if self._on_log:
            self._on_log(msg)

    def check_models(self) -> list[str]:
        symbols = self.config.get("model", "default_symbols", default=["XAUUSD"])
        missing = [s for s in symbols if not self._inference.validate_symbol_models(s)]
        return missing

    def connect_mt5(self) -> bool:
        if not self._bridge.initialize():
            self.log("MT5 连接失败")
            return False
        self.log("MT5 已连接")
        return True

    def start_runtime(self, on_signal: Callable) -> None:
        if self._bar_thread:
            return
        pipe = self.config.get("pipes", "data_pipe")
        self._bar_thread = threading.Thread(target=self._consume_bars, daemon=True, name="BarConsumer")
        self._bar_thread.start()

        threads = [
            DataReceiverThread(pipe, self._bar_queue, self._stop),
            SignalSchedulerThread(
                self.config,
                self._bar_store,
                self._inference,
                self._signal_gen,
                self._drawing,
                self._pending,
                self._pending_lock,
                on_signal,
                self._stop,
            ),
            PositionManagerThread(
                self.config, self._bridge, self._pending, self._pending_lock, self._stop
            ),
            MacroCalendarThread(
                float(self.config.get("macro", "reload_interval_hours", default=24)),
                self._stop,
            ),
        ]
        for t in threads:
            t.start()
            self._threads.append(t)
        self.log("烛龙运行时线程已启动")

    def _consume_bars(self) -> None:
        while not self._stop.is_set():
            try:
                bar = self._bar_queue.get(timeout=1.0)
            except Empty:
                continue
            self._bar_store.ingest_m1(bar)

    def stop(self) -> None:
        self._stop.set()
        self._bridge.shutdown()
        self.log("烛龙已停止")
