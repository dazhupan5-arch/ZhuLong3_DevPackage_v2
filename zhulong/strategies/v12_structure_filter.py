"""V12 概率 + 30 维结构特征过滤器（无需重训模型）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from zhulong.agent.structure_analyzer import FEATURE_NAMES


@dataclass
class V12StructureFilterConfig:
    enabled: bool = False
    long_prob_threshold: float = 0.65
    short_prob_threshold: float = 0.60
    min_support_strength: float = 0.4
    min_resistance_strength: float = 0.4
    max_support_dist: float = 0.5
    max_resistance_dist: float = 0.5
    require_breakout_confirm: bool = False
    require_divergence: bool = False
    mtf_align_min: float = 0.05
    allowed_hours: list[int] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> V12StructureFilterConfig:
        if not d:
            return cls()
        hours = d.get("allowed_hours")
        if hours is not None:
            hours = [int(h) for h in hours]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class V12WithStructureFilter:
    """
    将 V12 的概率输出与结构分析器（30 维）结合，只在关键位置开仓。
    不持有模型；调用方传入 predict_proba 结果即可。
    """

    def __init__(
        self,
        long_prob_threshold: float = 0.65,
        short_prob_threshold: float = 0.60,
        min_support_strength: float = 0.4,
        min_resistance_strength: float = 0.4,
        max_support_dist: float = 0.5,
        max_resistance_dist: float = 0.5,
        require_breakout_confirm: bool = False,
        require_divergence: bool = False,
        mtf_align_min: float = 0.05,
        allowed_hours: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs.pop("v12_model", None)
        _ = kwargs.pop("enabled", None)
        if kwargs:
            raise TypeError(f"未知参数: {', '.join(kwargs)}")
        self.long_th = float(long_prob_threshold)
        self.short_th = float(short_prob_threshold)
        self.min_support_strength = float(min_support_strength)
        self.min_resistance_strength = float(min_resistance_strength)
        self.max_support_dist = float(max_support_dist)
        self.max_resistance_dist = float(max_resistance_dist)
        self.require_breakout_confirm = bool(require_breakout_confirm)
        self.require_divergence = bool(require_divergence)
        self.mtf_align_min = float(mtf_align_min)
        self.allowed_hours = allowed_hours

    @classmethod
    def from_config(cls, cfg: V12StructureFilterConfig) -> V12WithStructureFilter:
        return cls(
            long_prob_threshold=cfg.long_prob_threshold,
            short_prob_threshold=cfg.short_prob_threshold,
            min_support_strength=cfg.min_support_strength,
            min_resistance_strength=cfg.min_resistance_strength,
            max_support_dist=cfg.max_support_dist,
            max_resistance_dist=cfg.max_resistance_dist,
            require_breakout_confirm=cfg.require_breakout_confirm,
            require_divergence=cfg.require_divergence,
            mtf_align_min=cfg.mtf_align_min,
            allowed_hours=cfg.allowed_hours,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> V12WithStructureFilter:
        return cls.from_config(V12StructureFilterConfig.from_dict(d))

    def _check_hour(self, dt: datetime | np.datetime64 | Any | None) -> bool:
        if self.allowed_hours is None or dt is None:
            return True
        if hasattr(dt, "hour"):
            return int(dt.hour) in self.allowed_hours
        return True

    @staticmethod
    def _feat_dict(feat_30d: np.ndarray) -> dict[str, float]:
        arr = np.asarray(feat_30d, dtype=np.float32).reshape(-1)
        n = min(len(arr), len(FEATURE_NAMES))
        return {FEATURE_NAMES[i]: float(arr[i]) for i in range(n)}

    def _mtf_aligned(self, f: dict[str, float]) -> bool:
        return f.get("mtf_trend_align", 0.0) >= self.mtf_align_min

    def _breakout_ok_long(self, f: dict[str, float]) -> bool:
        if not self.require_breakout_confirm:
            return True
        return f.get("m5_resistance_dist", 99.0) <= 0.35

    def _breakout_ok_short(self, f: dict[str, float]) -> bool:
        if not self.require_breakout_confirm:
            return True
        return f.get("m5_support_dist", 99.0) <= 0.35

    def should_open_long(
        self,
        feat_30d: np.ndarray,
        v12_prob_long: float,
        atr: float,
        close: float,
        dt: datetime | Any | None = None,
    ) -> bool:
        _ = atr, close
        if not self._check_hour(dt):
            return False
        if v12_prob_long < self.long_th:
            return False

        f = self._feat_dict(feat_30d)
        support_ok = (
            f.get("m5_support_dist", 99.0) <= self.max_support_dist
            and f.get("m5_support_strength", 0.0) >= self.min_support_strength
        )
        trend_ok = f.get("m5_trend", 0.0) > 0
        pattern_ok = f.get("double_bottom", 0.0) == 1.0 or f.get("inverse_head_shoulders", 0.0) == 1.0
        divergence_ok = f.get("rsi_bull_div", 0.0) == 1.0 or f.get("macd_bull_div", 0.0) == 1.0
        mtf_ok = self._mtf_aligned(f)
        breakout_ok = self._breakout_ok_long(f)

        if support_ok and trend_ok and mtf_ok and breakout_ok:
            return True
        if pattern_ok and support_ok and breakout_ok:
            return True
        if self.require_divergence and divergence_ok and support_ok and breakout_ok:
            return True
        return False

    def should_open_short(
        self,
        feat_30d: np.ndarray,
        v12_prob_short: float,
        atr: float,
        close: float,
        dt: datetime | Any | None = None,
    ) -> bool:
        _ = atr, close
        if not self._check_hour(dt):
            return False
        if v12_prob_short < self.short_th:
            return False

        f = self._feat_dict(feat_30d)
        resistance_ok = (
            f.get("m5_resistance_dist", 99.0) <= self.max_resistance_dist
            and f.get("m5_resistance_strength", 0.0) >= self.min_resistance_strength
        )
        trend_ok = f.get("m5_trend", 0.0) < 0
        pattern_ok = f.get("double_top", 0.0) == 1.0 or f.get("head_shoulders_top", 0.0) == 1.0
        divergence_ok = f.get("rsi_bear_div", 0.0) == 1.0 or f.get("macd_bear_div", 0.0) == 1.0
        mtf_ok = self._mtf_aligned(f)
        breakout_ok = self._breakout_ok_short(f)

        if resistance_ok and trend_ok and mtf_ok and breakout_ok:
            return True
        if pattern_ok and resistance_ok and breakout_ok:
            return True
        if self.require_divergence and divergence_ok and resistance_ok and breakout_ok:
            return True
        return False

    def get_signal(
        self,
        feat_30d: np.ndarray,
        v12_proba: np.ndarray | list[float],
        atr: float,
        close: float,
        dt: datetime | Any | None = None,
    ) -> int:
        """
        v12_proba: [flat, long, short] 三类概率。
        返回: 1=做多, -1=做空, 0=观望
        """
        _ = atr, close
        proba = np.asarray(v12_proba, dtype=np.float64).reshape(-1)
        if proba.size == 3:
            prob_long, prob_short = float(proba[1]), float(proba[2])
        elif proba.size == 2:
            prob_long, prob_short = float(proba[0]), float(proba[1])
        else:
            return 0

        if self.should_open_long(feat_30d, prob_long, atr, close, dt):
            return 1
        if self.should_open_short(feat_30d, prob_short, atr, close, dt):
            return -1
        return 0

    def reject_reason(
        self,
        feat_30d: np.ndarray,
        v12_proba: np.ndarray | list[float],
        atr: float,
        close: float,
        dt: datetime | Any | None = None,
    ) -> str:
        """供日志/调试：说明为何未开仓。"""
        proba = np.asarray(v12_proba, dtype=np.float64).reshape(-1)
        if proba.size == 3:
            p_long, p_short = float(proba[1]), float(proba[2])
        elif proba.size == 2:
            p_long, p_short = float(proba[0]), float(proba[1])
        else:
            return "invalid_proba"

        if not self._check_hour(dt):
            return "hour_filter"
        if p_long >= self.long_th and self.should_open_long(feat_30d, p_long, atr, close, dt):
            return ""
        if p_short >= self.short_th and self.should_open_short(feat_30d, p_short, atr, close, dt):
            return ""
        if max(p_long, p_short) < min(self.long_th, self.short_th):
            return f"prob<{self.long_th}/{self.short_th}"
        return "structure_filter"
