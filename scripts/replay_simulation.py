#!/usr/bin/env python3
"""
烛龙3 逐K线回放模拟器

从CSV文件读取历史M1数据，一根一根喂给烛龙全链路管道：
  M1 → FeatureCache(M1合成M5) → AI推理 → 信号生成 → 持仓管理(SL/TP/移动止损/时间停止)

使用方法:
  1. 先用 export_m1_data.mq5 导出数据到 CSV
  2. 将 CSV 放到 simulation/data/ 目录
  3. 运行本脚本: python replay_simulation.py --csv simulation/data/M1_export.csv

输出:
  - 控制台实时日志（模拟实盘运行状态）
  - simulation/output/replay_report.txt 完整报告
  - simulation/output/trade_log.csv 交易明细
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------- 路径设置 ----------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------- 导入烛龙模块 ----------
from zhulong.strategies.indicators import atr_series
from zhulong.utils.time_index import normalize_m5_index

os.environ["ZHULONG_IMF_CSV_ONLY"] = "1"

# =========================================================================
# 配置
# =========================================================================
CONFIG = {
    "signal_expiry_minutes": 240,
    "max_hold_minutes": 240,
    "trailing_activation_pct": 0.30,
    "trailing_step_pct": 0.15,
    "trailing_tighten_factor": 0.8,
    "partial_target1_pct": 0.25,
    "partial_target2_pct": 0.40,
    "max_drawdown_ratio": 0.7,
    "peak_profit_min_for_drawdown": 0.3,  # 峰值至少 0.3% 才启用回撤保护
    "cooldown_minutes": 30,
    "speed_multiplier": 100,
    "atr_period": 14,
}


# =========================================================================
# 数据结构
# =========================================================================
@dataclass
class SimulatedPosition:
    """模拟的持仓状态。"""
    signal_id: str
    symbol: str
    direction: str           # "buy" or "sell"
    entry_price: float
    open_time: float          # Unix 秒
    volume: float = 1.0
    # SL/TP
    stop_loss: float = 0.0
    take_profit: float = 0.0
    # 跟踪
    trailing_sl: float = 0.0
    trailing_activated: bool = False
    last_trail_price: float = 0.0
    peak_profit_pct: float = 0.0
    time_expired: bool = False
    # 状态
    is_open: bool = True
    close_reason: str = ""
    close_price: float = 0.0
    close_time: float = 0.0
    pnl_pct: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.open_time if self.is_open else 0.0


@dataclass
class SimulatedFeatureCache:
    """模拟的 FeatureCache，从M1累积合成M5。"""
    symbol: str
    m1_bars: list = field(default_factory=list)       # 累积的 M1 bar dict
    m5_bars: list = field(default_factory=list)        # 合成的 M5 bar (OHLC)
    last_m1_time: int = 0
    last_m5_time: int = 0
    m5_count: int = 0

    def ingest(self, bar: dict) -> bool:
        """喂入一根 M1 K线。返回 True 表示生成了新 M5。"""
        self.m1_bars.append(bar)
        self.last_m1_time = bar["time_unix"]

        # 检查是否完成了一根 M5
        bar_time = datetime.fromtimestamp(bar["time_unix"], tz=timezone.utc)
        m5_boundary = bar_time.minute // 5 * 5
        m5_key = bar_time.replace(minute=m5_boundary, second=0, microsecond=0)
        m5_unix = int(m5_key.timestamp())

        if m5_unix > self.last_m5_time:
            # 用最近5根M1合成M5
            recent = [b for b in self.m1_bars
                      if b["time_unix"] > self.last_m5_time and b["time_unix"] <= m5_unix]
            if len(recent) >= 2:  # 至少需要2根M1
                m5_bar = {
                    "time_unix": m5_unix,
                    "time": m5_key,
                    "open": recent[0]["open"],
                    "high": max(b["high"] for b in recent),
                    "low": min(b["low"] for b in recent),
                    "close": recent[-1]["close"],
                    "volume": sum(b["volume"] for b in recent),
                }
                self.m5_bars.append(m5_bar)
                self.last_m5_time = m5_unix
                self.m5_count = len(self.m5_bars)
                return True
        return False

    def get_m5_df(self) -> pd.DataFrame:
        """返回 M5 的 DataFrame（用于 AI 推理）。"""
        if not self.m5_bars:
            return pd.DataFrame()
        rows = []
        for b in self.m5_bars:
            rows.append({
                "time": b["time"],
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b["volume"],
            })
        df = pd.DataFrame(rows).set_index("time").sort_index()
        return normalize_m5_index(df[["open", "high", "low", "close", "volume"]])


# =========================================================================
# 日志和报告
# =========================================================================
class SimulationLogger:
    """模拟日志记录器。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.trades: list[dict] = []
        self.signals: list[dict] = []
        self.events: list[dict] = []

    def log(self, msg: str, cat: str = "INFO"):
        """记录一行日志。"""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{cat}] {msg}"
        self.lines.append(line)
        print(line)

    def log_signal(self, sig: dict):
        """记录信号事件。"""
        self.signals.append(sig)
        self.log(f"信号: {sig.get('symbol','')} {sig.get('direction','')} "
                 f"conf={sig.get('confidence',0):.2f} entry={sig.get('entry_price',0):.2f} "
                 f"sl={sig.get('stop_loss',0):.2f} tp={sig.get('take_profit',0):.2f}",
                 cat="SIGNAL")

    def log_trade_open(self, pos: SimulatedPosition):
        """记录开仓。"""
        self.log(f"开仓: {pos.symbol} {pos.direction} entry={pos.entry_price:.2f} "
                 f"sl={pos.stop_loss:.2f} tp={pos.take_profit:.2f} signal={pos.signal_id}",
                 cat="TRADE")

    def log_trade_close(self, pos: SimulatedPosition):
        """记录平仓。"""
        self.trades.append({
            "signal_id": pos.signal_id,
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "close_price": pos.close_price,
            "pnl_pct": pos.pnl_pct,
            "close_reason": pos.close_reason,
            "open_time": datetime.fromtimestamp(pos.open_time).strftime("%Y-%m-%d %H:%M"),
            "close_time": datetime.fromtimestamp(pos.close_time).strftime("%Y-%m-%d %H:%M"),
            "hold_minutes": (pos.close_time - pos.open_time) / 60,
        })
        self.log(f"平仓: {pos.symbol} {pos.signal_id} pnl={pos.pnl_pct:.2f}% "
                 f"reason={pos.close_reason} hold={(pos.close_time-pos.open_time)/60:.1f}min",
                 cat="TRADE")

    def log_state(self, pos: SimulatedPosition | None, price: float, tick_bid: float, tick_ask: float):
        """记录持仓状态快照。"""
        if pos and pos.is_open:
            profit = self._profit_pct(price, pos)
            self.events.append({
                "time": datetime.now().isoformat(),
                "symbol": pos.symbol,
                "direction": pos.direction,
                "price": price,
                "bid": tick_bid,
                "ask": tick_ask,
                "profit_pct": profit,
                "sl": pos.stop_loss if not pos.trailing_activated else pos.trailing_sl,
                "tp": pos.take_profit,
                "trailing_activated": pos.trailing_activated,
                "age_seconds": pos.age_seconds,
            })

    def generate_report(self, total_bars: int, elapsed_real: float):
        """生成最终报告。"""
        report_path = self.output_dir / "replay_report.txt"
        trade_csv = self.output_dir / "trade_log.csv"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("烛龙3 逐K线回放模拟报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模拟参数: speed={CONFIG['speed_multiplier']}x, "
                    f"max_hold={CONFIG['max_hold_minutes']}min, "
                    f"trailing_activation={CONFIG['trailing_activation_pct']}%\n")
            f.write("=" * 60 + "\n\n")

            stats = self._compute_stats()
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")

            f.write("\n" + "-" * 60 + "\n")
            f.write("全部日志:\n")
            f.write("-" * 60 + "\n")
            for line in self.lines:
                f.write(line + "\n")

        # 导出交易 CSV
        if self.trades:
            import csv as csv_module
            with open(trade_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv_module.DictWriter(f, fieldnames=self.trades[0].keys())
                writer.writeheader()
                writer.writerows(self.trades)

        self.log(f"报告已生成: {report_path}")
        self.log(f"交易明细: {trade_csv}")
        self.log(f"总M1 K线数: {total_bars}")
        self.log(f"模拟运行时间: {elapsed_real:.1f}s")

        return stats

    def _compute_stats(self) -> dict:
        """计算统计摘要。"""
        total_trades = len(self.trades)
        win_trades = sum(1 for t in self.trades if t["pnl_pct"] > 0)
        loss_trades = sum(1 for t in self.trades if t["pnl_pct"] <= 0)
        total_pnl = sum(t["pnl_pct"] for t in self.trades)
        avg_win = sum(t["pnl_pct"] for t in self.trades if t["pnl_pct"] > 0) / max(win_trades, 1)
        avg_loss = sum(t["pnl_pct"] for t in self.trades if t["pnl_pct"] <= 0) / max(loss_trades, 1)

        return {
            "总交易次数": total_trades,
            "盈利次数": win_trades,
            "亏损次数": loss_trades,
            "胜率": f"{win_trades/max(total_trades,1)*100:.1f}%",
            "总盈亏%": f"{total_pnl:+.2f}%",
            "平均盈利%": f"{avg_win:+.2f}%",
            "平均亏损%": f"{avg_loss:+.2f}%",
            "信号总数": len(self.signals),
            "信号采纳数": len(self.trades),
            "开仓品种": set(t["symbol"] for t in self.trades) if self.trades else set(),
            "平仓原因分布": self._reason_distribution(),
        }

    def _reason_distribution(self) -> str:
        reasons = {}
        for t in self.trades:
            r = t["close_reason"]
            reasons[r] = reasons.get(r, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in reasons.items())

    @staticmethod
    def _profit_pct(price: float, pos: SimulatedPosition) -> float:
        if pos.entry_price <= 0:
            return 0.0
        if pos.direction == "buy":
            return (price - pos.entry_price) / pos.entry_price * 100
        return (pos.entry_price - price) / pos.entry_price * 100


# =========================================================================
# 核心模拟器
# =========================================================================
class ReplaySimulator:
    """逐K线回放模拟器。

    将历史 M1 数据逐根喂给 AI 管道，模拟完整的信号→持仓→平仓生命周期。
    """

    def __init__(self, output_dir: Path):
        self.logger = SimulationLogger(output_dir)
        self.cache: dict[str, SimulatedFeatureCache] = {}
        self.position: SimulatedPosition | None = None  # 单信号约束：同时只有一单
        self.config = CONFIG
        self.ai_engine = None  # AI 引擎（延迟初始化）
        self._primary_symbol = "XAUUSD"
        self._initialized_ai = False
        self._m5_counter = 0
        self._signal_cooldowns: dict[str, float] = {}

    def load_csv(self, csv_path: str) -> list[dict]:
        """从 CSV 加载 M1 K 线数据，按时间升序返回。"""
        bars = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    bar = {
                        "time_unix": int(row["time_unix"]),
                        "time_str": row["time_str"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                    bars.append(bar)
                except (ValueError, KeyError) as e:
                    self.logger.log(f"跳过无效行: {row} ({e})", cat="WARN")

        # 按时间排序
        bars.sort(key=lambda b: b["time_unix"])
        self.logger.log(f"已加载 {len(bars)} 根 M1 K线: "
                        f"{bars[0]['time_str']} ~ {bars[-1]['time_str']}")
        return bars

    def initialize_ai(self, symbol: str):
        """初始化 AI 引擎（TradingAgent）。"""
        try:
            from zhulong.agent.trading_agent import TradingAgent

            config_path = _PROJECT_ROOT / "config" / "config_agent.json"
            if config_path.exists():
                agent_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            else:
                self.logger.log("config_agent.json 不存在，使用默认配置", cat="WARN")
                agent_config = {"enabled": True, "primary_symbol": symbol}

            agent_config["primary_symbol"] = symbol
            # 禁用元学习（P1-3 规避）
            if "meta_learning" in agent_config:
                agent_config["meta_learning"]["enabled"] = False

            self.ai_engine = TradingAgent(agent_config, root=_PROJECT_ROOT)
            self._primary_symbol = symbol
            self._initialized_ai = True
            self.logger.log(f"AI 引擎已初始化: {symbol}", cat="SYSTEM")
        except Exception as e:
            self.logger.log(f"AI 引擎初始化失败: {e}\n{traceback.format_exc()}", cat="ERROR")
            self._initialized_ai = False

    def run_ai_inference(self, symbol: str, m5_df: pd.DataFrame) -> dict | None:
        """对当前 M5 数据运行 AI 推理，返回信号（如有）。"""
        if not self._initialized_ai or self.ai_engine is None:
            return None

        try:
            m5_by_symbol = {symbol: m5_df}
            results = self.ai_engine.tick_symbols(m5_by_symbol, [symbol], account=None)
            if not results:
                return None

            result = results[0]
            if result.get("skipped"):
                return None

            sig = result.get("signal")
            if sig is None or sig.get("direction") not in ("buy", "sell"):
                return None

            return {
                "signal_id": sig.get("signal_id", ""),
                "symbol": symbol,
                "direction": sig["direction"],
                "entry_price": sig.get("entry", 0),
                "stop_loss": sig.get("sl", 0),
                "take_profit": sig.get("tp", 0),
                "confidence": sig.get("confidence", 0),
                "strategy": sig.get("strategy", "rl_agent"),
                "ai_sl_price": sig.get("ai_sl_price", 0.0),
                "ai_tp_price": sig.get("ai_tp_price", 0.0),
                "exit_assessment": result.get("cognition", {}).get("exit_assessment", 0.0)
                    if "cognition" in result else 0.0,
                "cognition": result.get("cognition", {}),
            }
        except Exception as e:
            self.logger.log(f"AI 推理异常: {e}", cat="WARN")
            return None

    def check_cooldown(self, symbol: str, direction: str) -> bool:
        """检查冷却期。"""
        key = f"{symbol}_{direction}"
        last = self._signal_cooldowns.get(key, 0)
        if time.time() - last < self.config["cooldown_minutes"] * 60:
            return True  # 冷却中
        return False

    def adopt_signal(self, sig: dict) -> SimulatedPosition | None:
        """采纳信号，创建持仓。"""
        # P2-1: 单信号约束
        if self.position is not None and self.position.is_open:
            self.logger.log(f"单信号约束: 已有活跃持仓 {self.position.signal_id}, "
                           f"拒绝信号 {sig.get('signal_id','')}", cat="CONSTRAINT")
            return None

        direction = sig.get("direction", "")
        symbol = sig.get("symbol", "")
        entry = sig.get("entry_price", 0)
        sl = sig.get("stop_loss", 0)
        tp = sig.get("take_profit", 0)

        # ===== P0-2: 使用 AI SL/TP 优先 =====
        if sig.get("ai_sl_price", 0) > 0:
            if (direction == "buy" and sig["ai_sl_price"] < entry) or \
               (direction == "sell" and sig["ai_sl_price"] > entry):
                sl = sig["ai_sl_price"]
                self.logger.log(f"使用 AI 计算的止损价: {sl:.2f}", cat="AI_SLTP")
        if sig.get("ai_tp_price", 0) > 0:
            if (direction == "buy" and sig["ai_tp_price"] > entry) or \
               (direction == "sell" and sig["ai_tp_price"] < entry):
                tp = sig["ai_tp_price"]
                self.logger.log(f"使用 AI 计算的止盈价: {tp:.2f}", cat="AI_SLTP")
        # ===== 结束 =====

        pos = SimulatedPosition(
            signal_id=sig.get("signal_id", ""),
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            open_time=time.time(),
            stop_loss=sl,
            take_profit=tp,
        )
        self.position = pos
        self.logger.log_trade_open(pos)

        # 设置冷却
        key = f"{symbol}_{direction}"
        self._signal_cooldowns[key] = time.time()

        return pos

    def update_position(self, tick_bid: float, tick_ask: float):
        """每 Tick 更新持仓状态（SL/TP 检查、移动止损、时间停止）。"""
        pos = self.position
        if pos is None or not pos.is_open:
            return

        # ===== P0-2: 使用正确的 Bid/Ask =====
        price = tick_bid if pos.direction == "buy" else tick_ask
        price_for_sl = tick_bid if pos.direction == "buy" else tick_ask
        price_for_tp = tick_ask if pos.direction == "buy" else tick_bid
        # ===== 结束 =====

        profit = self._profit_pct(price, pos)
        pos.peak_profit_pct = max(pos.peak_profit_pct, profit)

        effective_sl = pos.trailing_sl if pos.trailing_activated else pos.stop_loss

        # --- SL/TP 检查（使用正确的方向价格）---
        if effective_sl > 0:
            if (pos.direction == "buy" and price_for_sl <= effective_sl) or \
               (pos.direction == "sell" and price_for_sl >= effective_sl):
                self._close_position(price_for_sl, "stop_loss" if not pos.trailing_activated else "trailing_stop")
                return

        if pos.take_profit > 0:
            if (pos.direction == "buy" and price_for_tp >= pos.take_profit) or \
               (pos.direction == "sell" and price_for_tp <= pos.take_profit):
                self._close_position(price_for_tp, "take_profit")
                return

        # --- P0-1: 时间停止（到期且未盈利才平仓）---
        age = time.time() - pos.open_time
        if age >= self.config["max_hold_minutes"] * 60 and profit <= 0:
            self._close_position(price, "time_stop")
            self.logger.log(f"时间停止: 持仓{self.config['max_hold_minutes']}分未盈利, 平仓", cat="TIME_STOP")
            return
        if age >= self.config["max_hold_minutes"] * 60 and profit > 0:
            if not pos.time_expired:
                pos.time_expired = True
                self.logger.log(f"到期但盈利, 等待AI决策: profit={profit:.2f}%", cat="TIME_STOP")
        # ===== 结束 =====

        # --- Drawdown 保护（需峰值 > 指定阈值才生效）---
        if pos.peak_profit_pct > self.config["peak_profit_min_for_drawdown"] and \
           profit < pos.peak_profit_pct * (1 - self.config["max_drawdown_ratio"]):
            self._close_position(price, "trailing")
            return

        # --- 移动止损 ---
        if profit >= self.config["trailing_activation_pct"]:
            self._apply_trailing(pos, price)

        if pos.peak_profit_pct > 0:
            self.logger.log(f"持仓: {pos.symbol} {pos.direction} "
                           f"profit={profit:.2f}%  price={price:.2f}  "
                           f"sl={effective_sl:.2f}  tp={pos.take_profit:.2f}  "
                           f"age={age/60:.1f}min",
                           cat="POSITION")

    def _apply_trailing(self, pos: SimulatedPosition, price: float):
        """移动止损逻辑。"""
        if not pos.trailing_activated:
            pos.trailing_activated = True
            pos.trailing_sl = pos.entry_price
            pos.last_trail_price = pos.entry_price
            self.logger.log(f"移动止损激活(保本): SL={pos.entry_price:.2f}", cat="TRAIL")
            return

        dir_sign = 1.0 if pos.direction == "buy" else -1.0
        price_move = (price - pos.last_trail_price) * dir_sign
        step = self.config["trailing_step_pct"] * pos.entry_price / 100.0
        if price_move < step:
            return

        tighten = step * self.config["trailing_tighten_factor"]
        new_sl = pos.trailing_sl + tighten * dir_sign
        if abs(new_sl - pos.trailing_sl) <= 0.01:
            return

        old_sl = pos.trailing_sl
        pos.trailing_sl = new_sl
        pos.last_trail_price = price
        self.logger.log(f"移动止损: SL {old_sl:.2f} → {new_sl:.2f}", cat="TRAIL")

    def _close_position(self, close_price: float, reason: str):
        """平仓。"""
        if self.position is None or not self.position.is_open:
            return

        self.position.is_open = False
        self.position.close_price = close_price
        self.position.close_time = time.time()
        self.position.pnl_pct = self._profit_pct(close_price, self.position)
        self.position.close_reason = reason

        # P0-2: 记录正确的平仓价格
        self.logger.log(f"平仓价格: {close_price:.2f} (reason={reason})", cat="CLOSE")
        self.logger.log_trade_close(self.position)

    def run(self, bars: list[dict]):
        """运行主模拟循环。

        逐根喂 M1 K线 → 合成 M5 → AI 推理 → 信号采纳 → 持仓管理
        """
        total_bars = len(bars)
        start_real = time.time()
        sim_duration = bars[-1]["time_unix"] - bars[0]["time_unix"]
        self.logger.log(f"模拟开始: {total_bars} 根 M1 K线, "
                        f"时间跨度 {(sim_duration/3600):.1f} 小时, "
                        f"加速倍率 {self.config['speed_multiplier']}x",
                        cat="SYSTEM")

        # 初始化品种缓存
        symbol = bars[0].get("symbol", "XAUUSD")
        if symbol not in self.cache:
            self.cache[symbol] = SimulatedFeatureCache(symbol=symbol)

        # 预加载历史数据（前20根M1用于计算指标）
        preload_bars = 20
        preload = bars[:preload_bars]
        main_bars = bars[preload_bars:]

        for b in preload:
            self.cache[symbol].ingest(b)

        self.logger.log(f"预热完成: 已加载 {preload_bars} 根 M1", cat="SYSTEM")

        # 初始化 AI（需要 M5 数据就绪）
        if not self._initialized_ai:
            self.initialize_ai(symbol)

        # 主循环：逐根喂 M1
        for idx, bar in enumerate(main_bars):
            bar_idx = preload_bars + idx

            # 模拟 Tick 价格（用 M1 的 OHLC 近似）
            tick_bid = bar["close"]
            tick_ask = bar["close"] * 1.0001  # 近似 spread

            # 喂入 M1
            new_m5 = self.cache[symbol].ingest(bar)

            # 新 M5 生成时触发 AI 推理
            if new_m5:
                self._m5_counter += 1
                m5_df = self.cache[symbol].get_m5_df()
                if m5_df is not None and len(m5_df) >= 20:
                    # ---- 运行 AI 推理 ----
                    sig = self.run_ai_inference(symbol, m5_df)
                    if sig is not None:
                        self.logger.log_signal(sig)
                        # 检查冷却
                        if not self.check_cooldown(sig["symbol"], sig["direction"]):
                            # P2-1: 单信号约束由 adopt_signal 检查
                            pos = self.adopt_signal(sig)
                            if pos is not None:
                                self.logger.log(f"信号采纳: {sig['signal_id']}", cat="SIGNAL")
                        else:
                            self.logger.log(f"冷却中: {sig['symbol']}_{sig['direction']}", cat="COOLDOWN")

            # 每根 M1 更新持仓状态（模拟 500ms tick）
            self.update_position(tick_bid, tick_ask)

            # 进度条（每10% 或每分钟输出）
            progress = bar_idx * 100 // total_bars
            if progress % 10 == 0 and bar_idx > 0 and bar_idx % (total_bars // 10) < 2:
                elapsed = time.time() - start_real
                eta = elapsed / max(bar_idx - preload_bars, 1) * (total_bars - bar_idx)
                self.logger.log(f"进度: {progress}%  ({bar_idx}/{total_bars})  "
                               f"运行{elapsed:.0f}s  剩余约{eta:.0f}s",
                               cat="PROGRESS")

            # 时间加速（等待一小段时间模拟实时感，但加速）
            time.sleep(0.001)  # 1ms per bar = 实际1000x加速

        elapsed_real = time.time() - start_real
        self.logger.log(f"模拟结束: 运行时间 {elapsed_real:.1f}s", cat="SYSTEM")

        # 收盘前强制平仓
        if self.position and self.position.is_open:
            last_price = bars[-1]["close"]
            self._close_position(last_price, "session_end")
            self.logger.log(f"模拟结束: 持仓强制平仓 price={last_price:.2f}", cat="CLOSE")

        # 生成报告
        return self.logger.generate_report(total_bars, elapsed_real)

    @staticmethod
    def _profit_pct(price: float, pos: SimulatedPosition) -> float:
        if pos.entry_price <= 0:
            return 0.0
        if pos.direction == "buy":
            return (price - pos.entry_price) / pos.entry_price * 100
        return (pos.entry_price - price) / pos.entry_price * 100


# =========================================================================
# 入口
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="烛龙3 逐K线回放模拟器")
    parser.add_argument("--csv", default="", help="M1 CSV 文件路径")
    parser.add_argument("--symbol", default="XAUUSD", help="交易品种")
    parser.add_argument("--speed", type=int, default=100, help="时间加速倍率")
    parser.add_argument("--output", default="simulation/output", help="输出目录")
    args = parser.parse_args()

    # 如果未指定 CSV，尝试自动查找
    csv_path = args.csv
    if not csv_path:
        candidates = [
            Path("simulation/data/M1_export.csv"),
            _PROJECT_ROOT / "simulation" / "data" / "M1_export.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = str(c)
                break

    if not csv_path or not Path(csv_path).exists():
        print(f"错误: CSV 文件不存在。请使用 export_m1_data.mq5 导出数据。")
        print(f"使用方法: python {__file__} --csv path/to/M1_export.csv")
        sys.exit(1)

    CONFIG["speed_multiplier"] = args.speed
    output_dir = _PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将日志也写入文件
    log_path = output_dir / "replay_log.txt"
    sys.stdout = open(log_path, "w", encoding="utf-8")

    simulator = ReplaySimulator(output_dir)
    bars = simulator.load_csv(str(csv_path))

    if not bars:
        print("错误: 没有可用的 K 线数据")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"烛龙3 逐K线回放模拟启动")
    print(f"{'='*60}")
    print(f"CSV文件: {csv_path}")
    print(f"K线数量: {len(bars)}")
    print(f"时间范围: {bars[0]['time_str']} ~ {bars[-1]['time_str']}")
    print(f"加速倍率: {CONFIG['speed_multiplier']}x")
    print(f"最大持仓: {CONFIG['max_hold_minutes']}min")
    print(f"移动止损: {CONFIG['trailing_activation_pct']}%激活")
    print(f"{'='*60}\n")

    stats = simulator.run(bars)

    print(f"\n{'='*60}")
    print(f"模拟完成")
    print(f"{'='*60}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n完整报告: {output_dir / 'replay_report.txt'}")
    print(f"交易明细: {output_dir / 'trade_log.csv'}")


if __name__ == "__main__":
    main()
