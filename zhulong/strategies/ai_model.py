"""XGBoost V14 策略封装（XAUUSD / USOIL）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zhulong.strategies.base import BaseStrategy, StrategyContext, StrategySignal

logger = logging.getLogger(__name__)


class AIModelStrategy(BaseStrategy):
    name = "ai_model"

    def __init__(self, config: dict[str, Any] | None = None, root: Path | None = None) -> None:
        super().__init__(config)
        self._root = root or Path.cwd()
        self._engines: dict[str, Any] = {}

    def _sym_cfg(self, symbol: str, context: StrategyContext) -> dict:
        for src in (self.config.get("symbols"), context.config.get("symbols")):
            if isinstance(src, dict) and symbol in src:
                return dict(src[symbol])
        default_kind = "oil_v14" if symbol.upper() in ("USOIL", "WTI", "CL") else "xau_v14"
        return {"training_symbol": symbol, "broker_symbol": symbol, "kind": default_kind}

    def _get_engine(self, symbol: str, context: StrategyContext):
        if symbol in self._engines:
            return self._engines[symbol]
        sym_cfg = self._sym_cfg(symbol, context)
        kind = sym_cfg.get("kind", "xau_v14")
        if kind in ("oil_v14", "xau_v14", "oil_v1"):
            if kind == "oil_v1":
                from zhulong.inference.oil_v1 import OilV1Config, OilV1Inference

                cfg = OilV1Config.from_dict(sym_cfg)
                cfg.symbol = sym_cfg.get("training_symbol", symbol)
                cfg.broker_symbol = sym_cfg.get("broker_symbol", symbol)
                eng = OilV1Inference(cfg, root=self._root)
                eng.load()
                self._engines[symbol] = (eng, kind, sym_cfg)
                return self._engines[symbol]

            from zhulong.v14_live import load_v14_bundle

            bundle = load_v14_bundle(
                sym_cfg.get("training_symbol", symbol),
                model_subdir="v14",
                root=self._root,
            )
            self._engines[symbol] = ("v14_bundle", kind, sym_cfg, bundle)
            return self._engines[symbol]

        raise ValueError(f"不支持的 AI 策略 kind={kind!r}，请使用 xau_v14 / oil_v14")

    def on_bar(self, symbol: str, context: StrategyContext) -> StrategySignal | None:
        m5 = context.get_m5(symbol)
        if m5 is None or len(m5) < 80:
            return self.flat(symbol, self.name, "M5 不足")

        bar_time = m5.index[-1]
        entry = self._get_engine(symbol, context)
        eng_or_bundle, kind, sym_cfg = entry[0], entry[1], entry[2]

        try:
            if kind in ("xau_v14", "oil_v14"):
                from zhulong.v14_live import predict_v14, build_live_v14_features

                row, _, _, feats_row = build_live_v14_features(
                    sym_cfg.get("training_symbol", symbol), m5=m5
                )
                bundle = entry[3]
                sig = predict_v14(bundle, row, m5, bar_time, feats_row)
            elif kind == "oil_v1":
                from zhulong.live_oil_features import build_live_oil_row

                row, _, _, _ = build_live_oil_row(
                    sym_cfg.get("training_symbol", symbol),
                    m5=m5,
                    broker_symbol=sym_cfg.get("broker_symbol", symbol),
                )
                sig = eng_or_bundle.build_signal(m5, row, bar_time)
            else:
                return self.flat(symbol, self.name, f"未知 kind={kind}")
        except Exception as ex:
            logger.warning("[%s] AI 推理失败: %s", symbol, ex)
            return self.flat(symbol, self.name, str(ex))

        if sig.direction == "flat":
            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction="flat",
                confidence=sig.confidence,
                entry=sig.entry,
                sl=sig.sl,
                tp=sig.tp,
                signal_id=sig.signal_id,
                reject_reason=sig.reject_reason or "threshold",
                broker_symbol=sym_cfg.get("broker_symbol", symbol),
            )

        return StrategySignal(
            strategy=self.name,
            symbol=symbol,
            direction=sig.direction,
            confidence=sig.confidence,
            entry=sig.entry,
            sl=sig.sl,
            tp=sig.tp,
            signal_id=sig.signal_id,
            broker_symbol=sym_cfg.get("broker_symbol", symbol),
            metadata={"probabilities": sig.probabilities},
        )

    def get_market_condition(self) -> str:
        return "trend"
