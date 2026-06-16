"""ATR 动态网格（低波动震荡）。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from zhulong.strategies.base import BaseStrategy, StrategyContext, StrategySignal
from zhulong.strategies.indicators import atr_series


class GridSystem(BaseStrategy):
    name = "grid_system"

    def on_bar(self, symbol: str, context: StrategyContext) -> StrategySignal | None:
        m5 = context.get_m5(symbol)
        if m5 is None or len(m5) < 100:
            return self.flat(symbol, self.name, "M5 不足")

        cfg = self.config
        spacing_mult = float(cfg.get("grid_spacing_atr_mult", 1.5))
        levels = int(cfg.get("grid_levels", 3))
        vol_per = float(cfg.get("volume_per_level", 0.01))
        max_active = int(cfg.get("max_active_grids", 5))
        pctile = int(cfg.get("low_volatility_atr_percentile", 20))
        sl_mult = float(cfg.get("sl_atr_mult", 0.8))
        tp_mult = float(cfg.get("tp_atr_mult", 1.2))

        atr = atr_series(m5).dropna()
        if len(atr) < 50:
            return self.flat(symbol, self.name, "ATR 历史不足")

        cur_atr = float(atr.iloc[-1])
        threshold = float(np.percentile(atr.values, pctile))
        margin = float(cfg.get("low_volatility_atr_margin", 1.0))
        if cur_atr > threshold * margin:
            return self.flat(symbol, self.name, "波动偏高，网格未激活")

        price = float(m5["close"].iloc[-1])
        active: list[dict] = self.state.setdefault(f"{symbol}:grids", [])
        active = [g for g in active if not g.get("filled")]
        self.state[f"{symbol}:grids"] = active

        if len(active) >= max_active:
            return self.flat(symbol, self.name, "网格已满")

        spacing = cur_atr * spacing_mult
        last_emit = self.state.get(f"{symbol}:last_grid_bar")
        bar_key = str(m5.index[-1])
        if last_emit == bar_key:
            return self.flat(symbol, self.name, "本 bar 已挂网格")

        for i in range(1, levels + 1):
            buy_price = price - spacing * i
            sell_price = price + spacing * i
            if any(abs(g["price"] - buy_price) < spacing * 0.2 for g in active):
                continue
            direction = "buy" if price > buy_price else "sell"
            entry = buy_price if direction == "buy" else sell_price
            if direction == "buy":
                sl, tp = entry - cur_atr * sl_mult, entry + cur_atr * tp_mult
            else:
                sl, tp = entry + cur_atr * sl_mult, entry - cur_atr * tp_mult

            grid = {"price": entry, "direction": direction, "filled": False}
            active.append(grid)
            self.state[f"{symbol}:grids"] = active
            self.state[f"{symbol}:last_grid_bar"] = bar_key
            self.state[f"{symbol}:last_grid_utc"] = datetime.now(timezone.utc).isoformat()

            sym_cfg = (context.config.get("symbols") or {}).get(symbol, {})
            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=direction,
                confidence=0.55,
                entry=entry,
                sl=sl,
                tp=tp,
                broker_symbol=sym_cfg.get("broker_symbol", symbol),
                metadata={
                    "action": "limit_order",
                    "grid_level": i,
                    "volume": vol_per,
                    "spacing_atr": spacing_mult,
                },
            )

        return self.flat(symbol, self.name, "无新网格位")

    def get_market_condition(self) -> str:
        return "range"
