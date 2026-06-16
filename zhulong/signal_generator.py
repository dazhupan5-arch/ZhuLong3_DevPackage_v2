"""信号生成与过滤（G6/G7/G1）。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from zhulong.config_loader import Config
from zhulong.feature_engine import current_atr_pct
from zhulong.macro_calendar import macro_features

logger = logging.getLogger(__name__)


@dataclass
class PendingSignal:
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    expected_return: float
    magic_number: int
    comment_hint: str
    created_at: int
    expiry_minutes: int = 240


class SignalGenerator:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._last_signal_time: dict[tuple[str, str], float] = {}

    @staticmethod
    def signal_id(symbol: str, direction: str, ts: int) -> str:
        return f"{time.strftime('%Y%m%d_%H%M', time.gmtime(ts))}_{symbol}_{direction}"

    @staticmethod
    def magic_number(signal_id: str) -> int:
        return hash(signal_id) & 0xFFFF or 1

    @staticmethod
    def comment_hint(signal_id: str, prefix: str = "ZhuLong") -> str:
        return f"{prefix}_{signal_id}"

    def _cooldown_ok(self, symbol: str, direction: str) -> bool:
        key = (symbol, direction)
        last = self._last_signal_time.get(key, 0)
        cooldown = self._config.get("signal_filters", "cooldown_minutes", default=30) * 60
        return (time.time() - last) >= cooldown

    def _macro_silence(self) -> bool:
        macro = self._config.get("macro", default={}) or {}
        if not macro.get("force_silence"):
            return False
        # 简化：由 macro_features hours_to_next / since 判定
        feats = macro_features()
        before = macro.get("silence_before_minutes", 30) / 60.0
        after = macro.get("silence_after_minutes", 15) / 60.0
        if feats[0] <= before or feats[2] <= after:
            return True
        return False

    def try_generate(
        self,
        symbol: str,
        m5_raw,
        m5_feat,
        seq: np.ndarray,
        hourly: np.ndarray,
        inference: dict,
    ) -> tuple[Optional[PendingSignal], str]:
        sf = self._config.get("signal_filters", default={}) or {}
        sg = self._config.get("signal_geometry", default={}) or {}
        direction = inference["direction"]
        if direction == 0:
            return None, "direction_flat"
        dir_str = "buy" if direction == 1 else "sell"
        confidence = inference["confidence"]
        if confidence < sf.get("prob_threshold", 0.6):
            return None, "low_confidence"
        expected_return = inference["expected_return"]
        if expected_return < sf.get("min_expected_return", 0.15):
            return None, "low_expected_return"

        atr_pct = current_atr_pct(m5_raw)
        if not (sf.get("min_volatility_atr", 0.2) <= atr_pct <= sf.get("max_volatility_atr", 1.0)):
            return None, "volatility_filter"

        risk = atr_pct * 1.2
        if risk > 0 and expected_return / risk < sf.get("min_risk_reward", 1.5):
            return None, "risk_reward"

        entry_offset = inference["entry_offset"]
        if direction == 1:
            if not (sf.get("entry_offset_buy_min", -0.3) <= entry_offset * 100 <= sf.get("entry_offset_buy_max", -0.05)):
                return None, "entry_offset"
        else:
            if not (sf.get("entry_offset_sell_min", 0.05) <= entry_offset * 100 <= sf.get("entry_offset_sell_max", 0.3)):
                return None, "entry_offset"

        if not self._cooldown_ok(symbol, dir_str):
            return None, "cooldown"
        if self._macro_silence():
            return None, "macro_silence"

        close = float(m5_raw["close"].iloc[-1])
        # ===== P0-2: 尝试获取实时 Tick 价格修正入场 =====
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                if direction == 1 and tick.ask > 0:  # 买入用 Ask
                    entry = tick.ask * (1 + entry_offset)
                elif direction == -1 and tick.bid > 0:  # 卖出用 Bid
                    entry = tick.bid * (1 + entry_offset)
                else:
                    entry = close * (1 + entry_offset)
            else:
                entry = close * (1 + entry_offset)
        except (ImportError, Exception):
            entry = close * (1 + entry_offset)
        # ===== 结束 =====
        atr_abs = float(close * atr_pct / 100.0)  # 用 close 计算 ATR（稳定）
        sl_mult = sg.get("initial_stop_loss_atr_mult", 1.2)
        tp_mult = sg.get("initial_take_profit_atr_mult", 2.0)
        if direction == 1:
            sl = entry - atr_abs * sl_mult
            tp = entry + atr_abs * tp_mult
        else:
            sl = entry + atr_abs * sl_mult
            tp = entry - atr_abs * tp_mult

        now = int(time.time())
        sid = self.signal_id(symbol, dir_str, now)
        magic = self.magic_number(sid)
        prefix = self._config.get("mt5", "comment_prefix", default="ZhuLong")
        comment = self.comment_hint(sid, prefix)

        sig = PendingSignal(
            signal_id=sid,
            symbol=symbol,
            direction=dir_str,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            expected_return=expected_return,
            magic_number=magic,
            comment_hint=comment,
            created_at=now,
            expiry_minutes=int(sf.get("signal_expiry_minutes", 240)),
        )
        self._last_signal_time[(symbol, dir_str)] = time.time()
        return sig, "ok"

    def to_db_row(self, sig: PendingSignal, config: Config) -> dict:
        return {
            "signal_id": sig.signal_id,
            "timestamp": sig.created_at,
            "symbol": sig.symbol,
            "direction": sig.direction,
            "entry_price": sig.entry_price,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "confidence": sig.confidence,
            "expected_return": sig.expected_return,
            "magic_number": sig.magic_number,
            "comment_hint": sig.comment_hint,
            "status": "pending",
            "params_snapshot": json.dumps(config.raw, ensure_ascii=False),
            "created_at": sig.created_at,
        }
