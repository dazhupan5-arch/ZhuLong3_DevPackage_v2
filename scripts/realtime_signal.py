#!/usr/bin/env python3
"""
烛龙双品种实时推理服务（XAUUSD V14 + USOIL V14）：
- 每 5 分钟检测各品种新 M5 K 线（MT5 API）
- 计算特征 → V14 三分类推理 → 后处理
- 通过 ZhuLong_Drawing 管道发送 draw_signal（含 broker_symbol 供 MT5 图表过滤）

用法:
  py -3 scripts/realtime_signal.py
  py -3 scripts/realtime_signal.py --config-xau config/config_xau_v14.json --config-oil config/config_oil_v14.json
  py -3 scripts/realtime_signal.py --multi-strategy
  py -3 scripts/realtime_signal.py --scheduler
  py -3 scripts/realtime_signal.py --once
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401 — used in SymbolRunner v13 branch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.inference.oil_v1 import OilV1Config, OilV1Inference, load_oil_v1_config  # noqa: E402
from zhulong.live_oil_features import build_live_oil_row  # noqa: E402
from scripts.mt5_bridge import Mt5MarketData, PipeBridge  # noqa: E402
from zhulong.utils.paths import logs_dir  # noqa: E402

logger = logging.getLogger(__name__)


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


class SymbolRunner:
    """单品种推理循环。"""

    def __init__(self, name: str, kind: str, raw_cfg: dict) -> None:
        self.name = name
        self.kind = kind
        self.raw = raw_cfg
        self.broker_symbol = raw_cfg.get("broker_symbol") or raw_cfg.get("symbol") or name
        self.training_symbol = raw_cfg.get("training_symbol") or raw_cfg.get("symbol") or name
        self._last_bar: str | None = None
        self._mt5 = Mt5MarketData(self.training_symbol, broker_symbol=self.broker_symbol)
        self._m5_bars = int(raw_cfg.get("m5_bars", 2000))

        if kind == "xau_v14":
            from zhulong.v14_live import load_v14_bundle

            self._v14_bundle = load_v14_bundle(
                self.training_symbol, model_subdir="v14", root=_ROOT,
            )
            self._engine = None
        elif kind == "oil_v14":
            from zhulong.v14_live import load_v14_bundle

            self._v14_bundle = load_v14_bundle(
                self.training_symbol, model_subdir="v14", root=_ROOT,
            )
            self._engine = None
        elif kind == "oil_v1":
            self.cfg = OilV1Config.from_dict(raw_cfg)
            self.cfg.broker_symbol = self.broker_symbol
            self.cfg.symbol = self.training_symbol
            self._engine = OilV1Inference(self.cfg, root=_ROOT)
            self._engine.load()
        else:
            raise ValueError(f"不支持的品种 kind={kind!r}，请使用 xau_v14 / oil_v14")
        if kind not in ("xau_v14", "oil_v14") and self._engine is not None:
            state_bar = self._engine.state.last_m5_bar
            if state_bar and ":" in state_bar:
                sym, bar = state_bar.split(":", 1)
                if sym == self.training_symbol:
                    self._last_bar = bar
            elif state_bar:
                self._last_bar = state_bar

    def tick(self) -> dict | None:
        try:
            m5 = self._mt5.fetch_m5(self._m5_bars)
        except Exception as ex:
            logger.warning("[%s] M5 获取失败 (%s): %s", self.name, self.broker_symbol, ex)
            return None

        bar_time = m5.index[-1]
        bar_key = str(bar_time)
        if self._last_bar == bar_key:
            return None
        self._last_bar = bar_key
        logger.info("[%s] 新 M5 %s close=%.2f", self.name, bar_time, float(m5["close"].iloc[-1]))

        if self.kind in ("xau_v14", "oil_v14"):
            from zhulong.v14_live import predict_v14, build_live_v14_features

            row, _, _, feats_row = build_live_v14_features(
                self.training_symbol, m5=m5
            )
            sig = predict_v14(self._v14_bundle, row, m5, bar_time, feats_row)
        elif self.kind == "oil_v1":
            row, _, _, _ = build_live_oil_row(
                self.training_symbol, m5=m5, broker_symbol=self.broker_symbol
            )
            sig = self._engine.build_signal(m5, row, bar_time)
        else:
            raise RuntimeError(f"未知 kind={self.kind}")

        return {
            "symbol": self.name,
            "broker_symbol": self.broker_symbol,
            "time": bar_key,
            "direction": sig.direction,
            "confidence": sig.confidence,
            "reject_reason": sig.reject_reason,
            "signal_id": sig.signal_id,
            "signal": sig,
        }


class DualRealtimeSignalService:
    def __init__(
        self,
        config_xau: str | Path | None = None,
        config_oil: str | Path | None = None,
    ) -> None:
        xau_raw = load_json(config_xau or _ROOT / "config" / "config_xau_v14.json")
        oil_raw = load_json(config_oil or _ROOT / "config" / "config_oil_v14.json")

        xau_kind = "xau_v14" if xau_raw.get("feature_set", "v14") == "v14" else "xau_v14"
        oil_kind = "oil_v14" if oil_raw.get("feature_set", "v14") == "v14" else "oil_v1"

        self._poll = int(xau_raw.get("poll_seconds", oil_raw.get("poll_seconds", 30)))
        pipes = xau_raw.get("pipes") or oil_raw.get("pipes") or {}
        self._bridge = PipeBridge(
            data_pipe=pipes.get("data_pipe", r"\\.\pipe\ZhuLong_Data"),
            drawing_pipe=pipes.get("drawing_pipe", r"\\.\pipe\ZhuLong_Drawing"),
        )
        self._runners = [
            SymbolRunner("XAUUSD", xau_kind, xau_raw),
            SymbolRunner(oil_raw.get("training_symbol", "USOIL"), oil_kind, oil_raw),
        ]
        self._running = True
        self._expiry = int(xau_raw.get("signal_expiry_minutes", 240))

    def start_pipes(self) -> None:
        self._bridge.start()
        for r in self._runners:
            if not r._mt5.connect():
                logger.warning("[%s] MT5 未连接 broker=%s", r.name, r.broker_symbol)

    def stop(self) -> None:
        self._running = False
        self._bridge.stop()
        for r in self._runners:
            r._mt5.shutdown()

    def tick_all(self) -> list[dict]:
        results = []
        for runner in self._runners:
            try:
                r = runner.tick()
                if r is None:
                    continue
                sig = r["signal"]
                if sig.direction != "flat":
                    payload = sig.to_draw_payload(self._expiry)
                    if self._bridge.send_draw(payload):
                        logger.info(
                            "[%s] 信号已发送 %s %s entry=%.2f",
                            r["symbol"], sig.signal_id, sig.direction, sig.entry,
                        )
                        r["draw_sent"] = True
                    else:
                        logger.warning("[%s] 绘图管道未就绪: %s", r["symbol"], sig.signal_id)
                        r["draw_sent"] = False
                else:
                    logger.info("[%s] 观望: %s", r["symbol"], sig.reject_reason or "threshold")
                del r["signal"]
                results.append(r)
            except Exception as ex:
                logger.exception("[%s] tick 异常: %s", runner.name, ex)
        return results

    def run_loop(self) -> None:
        self.start_pipes()
        syms = ", ".join(f"{r.name}→{r.broker_symbol}" for r in self._runners)
        logger.info("双品种实时服务启动 [%s] poll=%ds (Ctrl+C 退出)", syms, self._poll)
        while self._running:
            self.tick_all()
            time.sleep(self._poll)

    def run_once(self) -> list[dict]:
        self.start_pipes()
        time.sleep(1.5)
        return self.tick_all()


class SchedulerRealtimeService:
    """自动调度模式：SchedulerCore + 动态权重 + 回撤保护。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        from zhulong.engine.scheduler_engine import SchedulerMt5Runner

        cfg = config_path or _ROOT / "config" / "config_scheduler.json"
        self._runner = SchedulerMt5Runner(cfg, root=_ROOT)
        self._poll = int(self._runner.config.get("poll_seconds", 30))
        self._running = True

    def start_pipes(self) -> None:
        self._runner.start()

    def stop(self) -> None:
        self._running = False
        self._runner.stop()

    def tick_all(self) -> list[dict]:
        return self._runner.tick_once()

    def run_loop(self) -> None:
        self.start_pipes()
        logger.info("自动调度服务启动 poll=%ds (Ctrl+C 退出)", self._poll)
        while self._running:
            for r in self.tick_all():
                sig = r.get("signal") or {}
                if sig.get("direction") in ("flat", None, ""):
                    logger.info(
                        "[%s] %s 观望 state=%s: %s",
                        r.get("symbol"),
                        r.get("strategy"),
                        r.get("state"),
                        sig.get("reject_reason") or r.get("risk", {}).get("block_reason") or "—",
                    )
                meta = (sig.get("metadata") or {})
                if meta.get("weights"):
                    logger.info("权重 %s", meta.get("weights"))
            time.sleep(self._poll)

    def run_once(self) -> list[dict]:
        self.start_pipes()
        time.sleep(1.5)
        return self.tick_all()


class MultiStrategyRealtimeService:
    """多策略模式：状态机调度 AI / 趋势 / 对冲 / 网格。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        from zhulong.engine.multi_strategy_engine import MultiStrategyMt5Runner

        cfg = config_path or _ROOT / "config" / "config_multi_strategy.json"
        self._runner = MultiStrategyMt5Runner(cfg, root=_ROOT)
        self._poll = int(self._runner.config.get("poll_seconds", 30))
        self._running = True

    def start_pipes(self) -> None:
        self._runner.start()

    def stop(self) -> None:
        self._running = False
        self._runner.stop()

    def tick_all(self) -> list[dict]:
        return self._runner.tick_once()

    def run_loop(self) -> None:
        self.start_pipes()
        logger.info("多策略实时服务启动 poll=%ds (Ctrl+C 退出)", self._poll)
        while self._running:
            for r in self.tick_all():
                sig = r.get("signal") or {}
                if sig.get("direction") == "flat":
                    logger.info(
                        "[%s] %s 观望 state=%s: %s",
                        r.get("symbol"),
                        r.get("strategy"),
                        r.get("state"),
                        sig.get("reject_reason") or "—",
                    )
            time.sleep(self._poll)

    def run_once(self) -> list[dict]:
        self.start_pipes()
        time.sleep(1.5)
        return self.tick_all()


class AgentRealtimeService:
    """RL 智能体模式；use_rl=false 时回退到 XGBoost v12 + USOIL v1。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        cfg_path = config_path or _ROOT / "config" / "config_agent.json"
        agent_cfg = load_json(cfg_path)
        self.use_rl = bool(agent_cfg.get("use_rl", False))
        if not self.use_rl:
            logger.info("use_rl=false，主策略回退 XGBoost v12 / USOIL v1")
            self._fallback = DualRealtimeSignalService()
            self._runner = None
            self._poll = self._fallback._poll
            self._running = True
            return

        from zhulong.engine.agent_engine import AgentMt5Runner

        self._fallback = None
        cfg = cfg_path
        self._runner = AgentMt5Runner(cfg, root=_ROOT)
        self._poll = int(self._runner.config.get("poll_seconds", 30))
        self._running = True

    def start_pipes(self) -> None:
        if self._fallback is not None:
            self._fallback.start_pipes()
            return
        self._runner.start()

    def stop(self) -> None:
        self._running = False
        if self._fallback is not None:
            self._fallback.stop()
            return
        self._runner.stop()

    def tick_all(self) -> list[dict]:
        if self._fallback is not None:
            return self._fallback.tick_all()
        return self._runner.tick_once()

    def run_loop(self) -> None:
        self.start_pipes()
        if self._fallback is not None:
            logger.info("智能体配置 use_rl=false，运行 XGBoost v12 双品种服务 poll=%ds", self._poll)
        else:
            logger.info("RL 智能体实时服务启动 poll=%ds (Ctrl+C 退出)", self._poll)
        while self._running:
            for r in self.tick_all():
                sig = r.get("signal") or {}
                if isinstance(sig, dict) and sig.get("direction") == "flat":
                    logger.info(
                        "[%s] %s action=%s: %s",
                        r.get("symbol"),
                        r.get("strategy"),
                        r.get("action"),
                        sig.get("reject_reason") or "—",
                    )
            time.sleep(self._poll)

    def run_once(self) -> list[dict]:
        self.start_pipes()
        time.sleep(1.5)
        return self.tick_all()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只跑一轮（测试）")
    parser.add_argument("--config-xau", default="config/config_xau_v14.json")
    parser.add_argument("--config-oil", default="config/config_oil_v14.json")
    parser.add_argument("--config", default="", help="兼容旧版：仅 XAU v12 单品种")
    parser.add_argument(
        "--multi-strategy",
        action="store_true",
        help="启用多策略引擎（config/config_multi_strategy.json）",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="启用自动调度引擎（config/config_scheduler.json，含动态权重+回撤保护）",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="启用智能体配置（config/config_agent.json；use_rl=false 时仍用 XGBoost v12）",
    )
    parser.add_argument("--agent-config", default="config/config_agent.json")
    parser.add_argument("--multi-config", default="config/config_multi_strategy.json")
    parser.add_argument("--scheduler-config", default="config/config_scheduler.json")
    args = parser.parse_args()

    log_dir = logs_dir()
    log_name = "trading.log"
    if args.scheduler:
        log_name = "scheduler.log"
    elif args.agent:
        log_name = "agent.log"
    elif args.multi_strategy:
        log_name = "multi_strategy.log"
    log_handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / log_name, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers,
    )

    if args.scheduler:
        svc = SchedulerRealtimeService(args.scheduler_config)
    elif args.agent:
        svc = AgentRealtimeService(args.agent_config)
    elif args.multi_strategy:
        svc = MultiStrategyRealtimeService(args.multi_config)
    elif args.config:
        svc = DualRealtimeSignalService(config_xau=args.config, config_oil=args.config)
        svc._runners = svc._runners[:1]
    else:
        svc = DualRealtimeSignalService(config_xau=args.config_xau, config_oil=args.config_oil)

    def _stop(*_):
        svc.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)

    if args.once:
        results = svc.run_once()
        print(json.dumps(results or [{"status": "no_new_bar"}], ensure_ascii=False, indent=2))
        svc.stop()
        return 0

    try:
        svc.run_loop()
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
