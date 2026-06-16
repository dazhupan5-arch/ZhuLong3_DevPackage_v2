"""USOIL v1 三分类推理 + EIA 屏蔽 + 极端趋势过滤。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.inference.signal_common import CooldownState, LiveSignal, resolve_path as _resolve
from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.oil_v1.backtest import (
    _eia_blackout_mask,
    h1_extreme_trend_filter,
)
from zhulong.training.v11.train import proba_to_directions
from zhulong.utils.paths import install_dir

logger = logging.getLogger(__name__)


@dataclass
class OilV1Config:
    symbol: str = "USOIL"
    training_symbol: str = "USOIL"
    broker_symbol: str = "USOIL"
    model_path: str = "models/USOIL/v1/xgb_triple_oil.json"
    meta_path: str = "models/USOIL/v1/oil_v1_meta.pkl"
    feature_columns: str = "data/training/oil_v1/USOIL/feature_columns.json"
    long_threshold: float = 0.86
    short_threshold: float = 0.84
    cooldown_minutes: int = 90
    cooldown_bars: int = 18
    long_sl_atr: float = 1.5
    short_sl_atr: float = 1.2
    tp_atr: float = 2.5
    use_eia_filter: bool = True
    eia_blackout_before_min: int = 30
    eia_blackout_after_min: int = 15
    max_daily_signals: int = 8
    min_atr_pct: float = 0.0015
    signal_expiry_minutes: int = 240
    state_file: str = "data/realtime_state_oil.json"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OilV1Config:
        mapped = dict(d)
        if "threshold_long" in mapped and "long_threshold" not in mapped:
            mapped["long_threshold"] = mapped["threshold_long"]
        if "threshold_short" in mapped and "short_threshold" not in mapped:
            mapped["short_threshold"] = mapped["threshold_short"]
        if "stop_loss_atr_mult_long" in mapped:
            mapped["long_sl_atr"] = mapped["stop_loss_atr_mult_long"]
        if "stop_loss_atr_mult_short" in mapped:
            mapped["short_sl_atr"] = mapped["stop_loss_atr_mult_short"]
        if "take_profit_atr_mult" in mapped:
            mapped["tp_atr"] = mapped["take_profit_atr_mult"]
        if "eia_blackout_before_minutes" in mapped:
            mapped["eia_blackout_before_min"] = mapped["eia_blackout_before_minutes"]
        if "eia_blackout_after_minutes" in mapped:
            mapped["eia_blackout_after_min"] = mapped["eia_blackout_after_minutes"]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapped.items() if k in known})


def load_oil_v1_config(path: str | Path | None = None) -> OilV1Config:
    root = install_dir()
    candidates = [
        Path(path) if path else None,
        root / "config" / "config_oil_v1.json",
        root / "models" / "USOIL" / "v1" / "config_oil_v1.json",
    ]
    for p in candidates:
        if p and p.is_file():
            return OilV1Config.from_dict(json.loads(p.read_text(encoding="utf-8")))
    return OilV1Config()


class OilV1Inference:
    def __init__(self, cfg: OilV1Config | None = None, root: Path | None = None) -> None:
        self.root = root or install_dir()
        self.cfg = cfg or load_oil_v1_config()
        self._model: xgb.XGBClassifier | None = None
        self._cols: list[str] = []
        self._state = CooldownState.load(_resolve(self.root, self.cfg.state_file))

    def load(self) -> None:
        model_p = _resolve(self.root, self.cfg.model_path)
        meta_p = _resolve(self.root, self.cfg.meta_path)
        cols_p = _resolve(self.root, self.cfg.feature_columns)
        if meta_p.is_file():
            meta = joblib.load(meta_p)
            self._cols = list(meta.get("feature_columns") or [])
            self.cfg.long_threshold = float(meta.get("long_threshold", self.cfg.long_threshold))
            self.cfg.short_threshold = float(meta.get("short_threshold", self.cfg.short_threshold))
        if not self._cols and cols_p.is_file():
            self._cols = json.loads(cols_p.read_text(encoding="utf-8"))
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_p))
        logger.info("oil v1 model loaded: %s (%d features)", model_p, len(self._cols))

    @property
    def state(self) -> CooldownState:
        return self._state

    def predict_proba(self, feature_row: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        return self._model.predict_proba(x)[0]

    def _in_eia_blackout(self, bar_time: pd.Timestamp) -> bool:
        if not self.cfg.use_eia_filter:
            return False
        mask = _eia_blackout_mask(pd.DatetimeIndex([bar_time]))
        return bool(mask[0])

    def apply_cooldown(self, direction: int, now: datetime) -> str | None:
        if direction == 0:
            return None
        day_key = f"{self.cfg.symbol}:{now.date().isoformat()}"
        if self._state.daily_counts.get(day_key, 0) >= self.cfg.max_daily_signals:
            return f"daily_limit={self.cfg.max_daily_signals}"
        last_key = "last_long_utc" if direction > 0 else "last_short_utc"
        last_iso = getattr(self._state, last_key)
        if last_iso:
            last = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
            mins = (now - last).total_seconds() / 60.0
            if mins < self.cfg.cooldown_minutes:
                return f"cooldown={mins:.0f}<{self.cfg.cooldown_minutes}min"
        return None

    def record_signal(self, direction: int, now: datetime, bar_time: pd.Timestamp) -> None:
        iso = now.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        day_key = f"{self.cfg.symbol}:{now.date().isoformat()}"
        self._state.daily_counts[day_key] = self._state.daily_counts.get(day_key, 0) + 1
        if direction > 0:
            self._state.last_long_utc = iso
        elif direction < 0:
            self._state.last_short_utc = iso
        self._state.last_m5_bar = f"{self.cfg.symbol}:{bar_time}"
        self._state.save(_resolve(self.root, self.cfg.state_file))

    def build_signal(
        self,
        m5: pd.DataFrame,
        feature_row: np.ndarray,
        bar_time: pd.Timestamp | None = None,
    ) -> LiveSignal:
        from zhulong.utils.time_index import normalize_m5_index, normalize_timestamp

        m5 = normalize_m5_index(m5)
        bar_time = normalize_timestamp(bar_time if bar_time is not None else m5.index[-1])
        broker = self.cfg.broker_symbol
        close = float(m5.loc[bar_time, "close"])
        atr_s = _atr_series(m5)
        idx = m5.index.get_loc(bar_time)
        if isinstance(idx, slice):
            idx = -1
        atr = float(atr_s.iloc[idx])
        if atr <= 0 or (atr / close) < self.cfg.min_atr_pct:
            return LiveSignal("flat", 0.0, close, 0.0, 0.0, "", broker, [], "atr_too_low")

        if self._in_eia_blackout(bar_time):
            return LiveSignal("flat", 0.0, close, 0.0, 0.0, "", broker, [], "eia_blackout")

        proba = self.predict_proba(feature_row)
        dirs = proba_to_directions(
            proba.reshape(1, -1),
            self.cfg.long_threshold,
            self.cfg.short_threshold,
        )
        dirs = h1_extreme_trend_filter(m5, pd.DatetimeIndex([bar_time]), dirs)
        direction = int(dirs[0])
        p0, p1, p2 = float(proba[0]), float(proba[1]), float(proba[2])
        conf = p1 if direction == 1 else (p2 if direction == -1 else max(p0, p1, p2))

        now = datetime.now(timezone.utc)
        cd = self.apply_cooldown(direction, now)
        if cd:
            return LiveSignal("flat", conf, close, 0.0, 0.0, "", broker, proba.tolist(), cd)
        if direction == 0:
            return LiveSignal("flat", conf, close, 0.0, 0.0, "", broker, proba.tolist(), "no_signal")

        side = "buy" if direction > 0 else "sell"
        sl_mult = self.cfg.long_sl_atr if direction > 0 else self.cfg.short_sl_atr
        if side == "buy":
            sl = close - atr * sl_mult
            tp = close + atr * self.cfg.tp_atr
        else:
            sl = close + atr * sl_mult
            tp = close - atr * self.cfg.tp_atr

        sig_id = f"{now.strftime('%Y%m%d_%H%M')}_{self.cfg.symbol}_{side}"
        self.record_signal(direction, now, bar_time)
        return LiveSignal(side, conf, close, sl, tp, sig_id, broker, proba.tolist(), "")
