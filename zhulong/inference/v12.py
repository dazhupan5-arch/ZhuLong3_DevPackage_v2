"""v12 三分类推理 + 后处理规则（冷却/趋势过滤/日限额）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.training.v11.train import proba_to_directions
from zhulong.training.v12.backtest import (
    V12_LONG_SL,
    V12_SHORT_SL,
    apply_short_trend_filter,
)
from zhulong.training.lgb.backtest import TP_ATR, _atr_series
from zhulong.training.v10.backtest import MIN_ATR_PCT
from zhulong.utils.paths import install_dir

logger = logging.getLogger(__name__)


@dataclass
class V12Config:
    symbol: str = "XAUUSD"
    model_path: str = "models/XAUUSD/xgb_triple.json"
    meta_path: str = "models/XAUUSD/v12_meta.pkl"
    feature_columns: str = "models/XAUUSD/feature_columns.json"
    long_threshold: float = 0.84
    short_threshold: float = 0.88
    long_cooldown_bars: int = 18
    short_cooldown_bars: int = 24
    long_cooldown_minutes: int = 90
    short_cooldown_minutes: int = 120
    max_daily_signals: int = 10
    long_sl_atr: float = V12_LONG_SL
    short_sl_atr: float = V12_SHORT_SL
    tp_atr: float = TP_ATR
    min_atr_pct: float = MIN_ATR_PCT
    signal_expiry_minutes: int = 240
    state_file: str = "data/v12_realtime_state.json"
    use_h1_trend_filter: bool = False
    structure_filter: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> V12Config:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        alias = {
            "cooldown_long_minutes": "long_cooldown_minutes",
            "cooldown_short_minutes": "short_cooldown_minutes",
            "stop_loss_atr_mult_long": "long_sl_atr",
            "stop_loss_atr_mult_short": "short_sl_atr",
            "take_profit_atr_mult": "tp_atr",
        }
        mapped: dict[str, Any] = {}
        for k, v in d.items():
            key = alias.get(k, k)
            if key in known:
                mapped[key] = v
        if "structure_filter" not in mapped and isinstance(d.get("structure_filter"), dict):
            mapped["structure_filter"] = d["structure_filter"]
        return cls(**mapped)


@dataclass
class V12Signal:
    direction: str  # buy | sell | flat
    confidence: float
    entry: float
    sl: float
    tp: float
    signal_id: str
    symbol: str
    probabilities: list[float]
    reject_reason: str = ""

    def to_draw_payload(self, expiry_minutes: int = 240) -> dict:
        if self.direction == "flat":
            return {}
        return {
            "action": "draw_signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": round(self.confidence, 4),
            "expiry_minutes": expiry_minutes,
        }


@dataclass
class CooldownState:
    last_long_utc: str | None = None
    last_short_utc: str | None = None
    last_m5_bar: str | None = None
    daily_counts: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "last_long_utc": self.last_long_utc,
                    "last_short_utc": self.last_short_utc,
                    "last_m5_bar": self.last_m5_bar,
                    "daily_counts": self.daily_counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CooldownState:
        if not path.is_file():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_long_utc=d.get("last_long_utc"),
                last_short_utc=d.get("last_short_utc"),
                last_m5_bar=d.get("last_m5_bar"),
                daily_counts={str(k): int(v) for k, v in (d.get("daily_counts") or {}).items()},
            )
        except Exception as ex:
            logger.warning("状态文件损坏，重置: %s", ex)
            return cls()


def load_v12_config(path: str | Path | None = None) -> V12Config:
    root = install_dir()
    candidates = [
        Path(path) if path else None,
        root / "config_v12.json",
        root / "models" / "XAUUSD" / "config_v12.json",
    ]
    for p in candidates:
        if p and p.is_file():
            return V12Config.from_dict(json.loads(p.read_text(encoding="utf-8")))
    return V12Config()


def _resolve(root: Path, p: str) -> Path:
    from zhulong.utils.paths import resolve_runtime_path

    return resolve_runtime_path(p, root=root)


class V12Inference:
    def __init__(self, cfg: V12Config | None = None, root: Path | None = None) -> None:
        self.root = root or install_dir()
        self.cfg = cfg or load_v12_config()
        self._model: xgb.XGBClassifier | None = None
        self._cols: list[str] = []
        self._state = CooldownState.load(_resolve(self.root, self.cfg.state_file))
        self._structure_analyzer = None
        self._structure_filter = None
        sf = self.cfg.structure_filter or {}
        if sf.get("enabled"):
            from zhulong.agent.structure_analyzer import StructureAnalyzer
            from zhulong.strategies.v12_structure_filter import V12WithStructureFilter

            sa_cfg = sf.get("analyzer") or {}
            self._structure_analyzer = StructureAnalyzer(sa_cfg)
            self._structure_filter = V12WithStructureFilter.from_dict(sf)
            logger.info("v12 structure filter enabled")

    def load(self) -> None:
        model_p = _resolve(self.root, self.cfg.model_path)
        if not model_p.is_file():
            alt = self.root / "models" / self.cfg.symbol / "v11" / "xgb_triple.json"
            if alt.is_file():
                model_p = alt
        cols_p = _resolve(self.root, self.cfg.feature_columns)
        self._cols = json.loads(cols_p.read_text(encoding="utf-8"))
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_p))
        logger.info("v12 model loaded: %s (%d features)", model_p, len(self._cols))

    @property
    def state(self) -> CooldownState:
        return self._state

    def predict_proba(self, feature_row: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        x = np.asarray(feature_row, dtype=np.float32).reshape(1, -1)
        return self._model.predict_proba(x)[0]

    def infer_direction(
        self,
        proba: np.ndarray,
        m5: pd.DataFrame,
        feats_row: pd.DataFrame,
        bar_time: pd.Timestamp,
    ) -> tuple[int, float]:
        """返回 (direction_int, confidence)，direction: 1 long, -1 short, 0 flat。"""
        dirs = proba_to_directions(
            proba.reshape(1, -1),
            self.cfg.long_threshold,
            self.cfg.short_threshold,
        )
        d = int(dirs[0])
        if d != 0 and self.cfg.use_h1_trend_filter:
            from zhulong.training.v13.triple import apply_trend_filter_v3
            filtered = apply_trend_filter_v3(m5, pd.DatetimeIndex([bar_time]), dirs)
            d = int(filtered[0])
        elif d < 0:
            filtered = apply_short_trend_filter(
                m5, feats_row, pd.DatetimeIndex([bar_time]), dirs
            )
            d = int(filtered[0])
        p0, p1, p2 = float(proba[0]), float(proba[1]), float(proba[2])
        if d == 1:
            conf = p1
        elif d == -1:
            conf = p2
        else:
            conf = max(p0, p1, p2)
        return d, conf

    def apply_cooldown(self, direction: int, now: datetime) -> str | None:
        if direction == 0:
            return None
        day_key = now.date().isoformat()
        if self._state.daily_counts.get(day_key, 0) >= self.cfg.max_daily_signals:
            return f"daily_limit={self.cfg.max_daily_signals}"
        if direction > 0 and self._state.last_long_utc:
            last = datetime.fromisoformat(self._state.last_long_utc.replace("Z", "+00:00"))
            mins = (now - last).total_seconds() / 60.0
            if mins < self.cfg.long_cooldown_minutes:
                return f"long_cooldown={mins:.0f}<{self.cfg.long_cooldown_minutes}min"
        if direction < 0 and self._state.last_short_utc:
            last = datetime.fromisoformat(self._state.last_short_utc.replace("Z", "+00:00"))
            mins = (now - last).total_seconds() / 60.0
            if mins < self.cfg.short_cooldown_minutes:
                return f"short_cooldown={mins:.0f}<{self.cfg.short_cooldown_minutes}min"
        return None

    def record_signal(self, direction: int, now: datetime, bar_time: pd.Timestamp) -> None:
        iso = now.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        day_key = now.date().isoformat()
        self._state.daily_counts[day_key] = self._state.daily_counts.get(day_key, 0) + 1
        if direction > 0:
            self._state.last_long_utc = iso
        elif direction < 0:
            self._state.last_short_utc = iso
        self._state.last_m5_bar = str(bar_time)
        self._state.save(_resolve(self.root, self.cfg.state_file))

    def build_signal(
        self,
        m5: pd.DataFrame,
        feature_row: np.ndarray,
        feats_df: pd.DataFrame,
        bar_time: pd.Timestamp | None = None,
    ) -> V12Signal:
        if bar_time is None:
            bar_time = m5.index[-1]
        close = float(m5.loc[bar_time, "close"])
        atr_s = _atr_series(m5)
        idx = m5.index.get_loc(bar_time)
        if isinstance(idx, slice):
            idx = -1
        atr = float(atr_s.iloc[idx])
        if pd.isna(atr) or atr <= 0 or (atr / close) < self.cfg.min_atr_pct:
            return V12Signal("flat", 0.0, close, 0.0, 0.0, "", self.cfg.symbol, [], "atr_too_low")

        proba = self.predict_proba(feature_row)
        reject = ""
        if self._structure_filter is not None and self._structure_analyzer is not None:
            feat_30d = self._structure_analyzer.compute_latest({"M5": m5})
            direction = self._structure_filter.get_signal(feat_30d, proba, atr, close, bar_time)
            p0, p1, p2 = float(proba[0]), float(proba[1]), float(proba[2])
            if direction == 1:
                conf = p1
            elif direction == -1:
                conf = p2
            else:
                conf = max(p0, p1, p2)
                reject = self._structure_filter.reject_reason(feat_30d, proba, atr, close, bar_time)
        else:
            direction, conf = self.infer_direction(proba, m5, feats_df, bar_time)
        now = datetime.now(timezone.utc)
        cd = self.apply_cooldown(direction, now)
        if cd:
            return V12Signal(
                "flat", conf, close, 0.0, 0.0, "", self.cfg.symbol,
                proba.tolist(), cd,
            )

        if direction == 0:
            reason = reject if self._structure_filter else "no_signal"
            return V12Signal("flat", conf, close, 0.0, 0.0, "", self.cfg.symbol, proba.tolist(), reason)

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
        return V12Signal(
            side, conf, close, sl, tp, sig_id, self.cfg.symbol, proba.tolist(), ""
        )
