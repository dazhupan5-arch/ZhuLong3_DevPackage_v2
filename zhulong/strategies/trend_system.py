"""EMA 多过滤器趋势系统（M5）。"""

from __future__ import annotations

from datetime import datetime, timezone

from zhulong.strategies.base import BaseStrategy, StrategyContext, StrategySignal
from zhulong.strategies.indicators import atr_series, ema, ema_cross_down, ema_cross_up


class TrendSystem(BaseStrategy):
    name = "trend_system"

    def on_bar(self, symbol: str, context: StrategyContext) -> StrategySignal | None:
        m5 = context.get_m5(symbol)
        if m5 is None or len(m5) < 60:
            return self.flat(symbol, self.name, "M5 不足")

        cfg = self.config
        fast_n = int(cfg.get("fast_ema", 9))
        slow_n = int(cfg.get("slow_ema", 21))
        trend_n = int(cfg.get("trend_ema", 50))
        sl_mult = float(cfg.get("sl_atr_mult", 1.0))
        tp_mult = float(cfg.get("tp_atr_mult", 1.5))
        min_atr_pct = float(cfg.get("min_atr_pct", 0.1))
        cooldown_min = int(cfg.get("cooldown_minutes", 60))
        max_daily = int(cfg.get("max_daily_signals", 0))

        close = m5["close"]
        fast = ema(close, fast_n)
        slow = ema(close, slow_n)
        trend = ema(close, trend_n)
        atr = atr_series(m5)
        atr_val = float(atr.iloc[-1])
        price = float(close.iloc[-1])
        if price <= 0 or atr_val <= 0:
            return self.flat(symbol, self.name, "ATR 无效")

        atr_p = atr_val / price * 100.0
        if atr_p < min_atr_pct:
            return self.flat(symbol, self.name, f"ATR% {atr_p:.3f} 过低")

        key = f"{symbol}:last_signal"
        last_ts = self.state.get(key)
        if last_ts:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_ts)).total_seconds() / 60
            if elapsed < cooldown_min:
                return self.flat(symbol, self.name, f"冷却 {cooldown_min}min")

        if max_daily > 0:
            day = datetime.now(timezone.utc).date().isoformat()
            daily_key = f"{symbol}:daily_signals"
            daily_counts: dict[str, int] = self.state.get(daily_key) or {}
            if daily_counts.get(day, 0) >= max_daily:
                return self.flat(symbol, self.name, f"daily_limit={max_daily}")

        trend_up = float(trend.iloc[-1]) < price
        trend_down = float(trend.iloc[-1]) > price

        direction = "flat"
        confidence = 0.7
        if ema_cross_up(fast, slow) and trend_up:
            direction = "buy"
        elif ema_cross_down(fast, slow) and trend_down:
            direction = "sell"

        if direction == "flat":
            return self.flat(symbol, self.name, "无金叉/死叉")

        if direction == "buy":
            sl = price - atr_val * sl_mult
            tp = price + atr_val * tp_mult
        else:
            sl = price + atr_val * sl_mult
            tp = price - atr_val * tp_mult

        now = datetime.now(timezone.utc)
        self.state[key] = now.isoformat()
        if max_daily > 0:
            day = now.date().isoformat()
            daily_key = f"{symbol}:daily_signals"
            daily_counts: dict[str, int] = dict(self.state.get(daily_key) or {})
            daily_counts[day] = daily_counts.get(day, 0) + 1
            self.state[daily_key] = daily_counts
        return StrategySignal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry=price,
            sl=sl,
            tp=tp,
        )

    def get_market_condition(self) -> str:
        return "trend"
