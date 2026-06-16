"""金油比 Z-score 跨品种对冲。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zhulong.strategies.base import BaseStrategy, StrategyContext, StrategySignal

logger = logging.getLogger(__name__)


class SpreadHedge(BaseStrategy):
    name = "spread_hedge"

    def on_bar(self, symbol: str, context: StrategyContext) -> StrategySignal | None:
        gold_sym = self.config.get("gold_symbol", "XAUUSD")
        oil_sym = self.config.get("oil_symbol", "USOIL")
        gold_m5 = context.get_m5(gold_sym)
        oil_m5 = context.get_m5(oil_sym)
        if gold_m5 is None or oil_m5 is None or len(gold_m5) < 50 or len(oil_m5) < 50:
            return self.flat(symbol, self.name, "金/油 M5 不足")

        lookback_days = int(self.config.get("lookback_days", 30))
        entry_z = float(self.config.get("entry_zscore", 2.0))
        exit_z = float(self.config.get("exit_zscore", 0.5))
        stop_z = float(self.config.get("stop_loss_zscore", 3.0))

        ratio_series = self._gold_oil_ratio(gold_m5, oil_m5, lookback_days)
        if ratio_series is None or len(ratio_series) < 20:
            return self.flat(symbol, self.name, "金油比对齐失败")

        mean = float(ratio_series.mean())
        std = float(ratio_series.std())
        if std <= 1e-9:
            return self.flat(symbol, self.name, "金油比波动过小")

        cur_ratio = float(ratio_series.iloc[-1])
        z = (cur_ratio - mean) / std
        self.state["last_z"] = z

        pos = self.state.get("pair_position")
        if pos == "short_gold_long_oil" and z <= exit_z:
            return self._flat_pair(gold_sym, "价差回归平空金多油")
        if pos == "long_gold_short_oil" and z >= -exit_z:
            return self._flat_pair(gold_sym, "价差回归平多金空油")

        gold_price = float(gold_m5["close"].iloc[-1])
        atr = context.get_atr(gold_sym)
        if atr <= 0:
            return self.flat(symbol, self.name, "ATR 无效")

        sym_cfg = (context.config.get("symbols") or {}).get(gold_sym, {})
        broker = sym_cfg.get("broker_symbol", gold_sym)

        if z > entry_z:
            self.state["pair_position"] = "short_gold_long_oil"
            return StrategySignal(
                strategy=self.name,
                symbol=gold_sym,
                direction="sell",
                confidence=min(0.95, 0.5 + abs(z) / 10),
                entry=gold_price,
                sl=gold_price + atr * 1.2,
                tp=gold_price - atr * 2.0,
                broker_symbol=broker,
                metadata={
                    "pair_action": "pair_short_gold_long_oil",
                    "zscore": round(z, 3),
                    "oil_symbol": oil_sym,
                    "stop_zscore": stop_z,
                },
            )
        if z < -entry_z:
            self.state["pair_position"] = "long_gold_short_oil"
            return StrategySignal(
                strategy=self.name,
                symbol=gold_sym,
                direction="buy",
                confidence=min(0.95, 0.5 + abs(z) / 10),
                entry=gold_price,
                sl=gold_price - atr * 1.2,
                tp=gold_price + atr * 2.0,
                broker_symbol=broker,
                metadata={
                    "pair_action": "pair_long_gold_short_oil",
                    "zscore": round(z, 3),
                    "oil_symbol": oil_sym,
                    "stop_zscore": stop_z,
                },
            )

        if abs(z) >= stop_z:
            return self.flat(symbol, self.name, f"Z={z:.2f} 超止损带，观望")

        return self.flat(symbol, self.name, f"Z={z:.2f} 未达入场")

    def _flat_pair(self, symbol: str, reason: str) -> StrategySignal:
        self.state.pop("pair_position", None)
        return self.flat(symbol, self.name, reason)

    @staticmethod
    def _gold_oil_ratio(
        gold_m5: pd.DataFrame, oil_m5: pd.DataFrame, lookback_days: int
    ) -> pd.Series | None:
        g = gold_m5["close"].resample("1D").last().dropna()
        o = oil_m5["close"].resample("1D").last().dropna()
        merged = pd.concat([g.rename("gold"), o.rename("oil")], axis=1).dropna()
        if merged.empty:
            idx = gold_m5.index.intersection(oil_m5.index)
            if len(idx) < 20:
                return None
            ratio = gold_m5.loc[idx, "close"] / oil_m5.loc[idx, "close"].replace(0, np.nan)
            return ratio.dropna().tail(lookback_days * 288)
        merged["ratio"] = merged["gold"] / merged["oil"].replace(0, np.nan)
        tail = merged["ratio"].dropna().tail(max(lookback_days, 5))
        if len(tail) < 5:
            return None
        daily = tail
        last_day = daily.index[-1]
        intraday = gold_m5.index[-1]
        if hasattr(intraday, "normalize") and last_day.normalize() == intraday.normalize():
            return daily
        return daily

    def get_market_condition(self) -> str:
        return "volatile"
