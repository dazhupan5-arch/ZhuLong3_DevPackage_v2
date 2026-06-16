"""L2 1 小时方向预测：对称规则，预测即方向。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch  # noqa: F401 — Windows: 须在 numpy 之前加载，避免 c10.dll 冲突

import numpy as np

from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.rl_agent import _resolve_model_path
from zhulong.agent.tick_brief import HorizonForecast, StructureSnapshot


def direction_from_probs(
    short_p: float,
    flat_p: float,
    long_p: float,
    *,
    min_confidence: float = 0.42,
) -> tuple[str, float, float, float]:
    """对称：long vs short 谁大跟谁；max(long,short) < 门槛 → flat。"""
    s, f, l = float(short_p), float(flat_p), float(long_p)
    total = max(s + f + l, 1e-9)
    s, f, l = s / total, f / total, l / total
    trade_conf = max(l, s)
    if trade_conf < min_confidence:
        return "flat", l, s, f
    if l > s:
        return "long", l, s, f
    if s > l:
        return "short", l, s, f
    return "flat", l, s, f


class HorizonPredictor:
    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
    ) -> None:
        self.root = root
        arch = config.get("architecture") or {}
        hp = arch.get("horizon_predictor") or {}
        self.horizon_bars = int(hp.get("horizon_bars", 12))
        self.gain_threshold = float(hp.get("gain_threshold", 0.002))
        self.min_confidence = float(hp.get("min_direction_confidence", 0.42))
        self.model_id = str(hp.get("model_id", "horizon_v16"))

        model_path = _resolve_model_path(str(hp.get("model_path", "models/horizon_v16.pth")), root)
        scaler_path = _resolve_model_path(str(hp.get("scaler_path", "models/horizon_v16_scaler.pkl")), root)
        onnx = model_path.with_suffix(".onnx")
        load_path = onnx if onnx.is_file() else model_path
        self._kn: KnowledgeNetInference | None = None
        if load_path.is_file():
            try:
                self._kn = KnowledgeNetInference(
                    load_path, scaler_path=scaler_path if scaler_path.is_file() else None, allow_pytorch=False
                )
            except Exception as ex:
                import logging
                logging.getLogger(__name__).warning("Horizon model load failed: %s", ex)
                self._kn = None

    @property
    def is_ready(self) -> bool:
        return self._kn is not None and self._kn.is_ready

    def predict(self, snapshot: StructureSnapshot) -> HorizonForecast:
        x = np.asarray(snapshot.vector, dtype=np.float32).reshape(1, -1)
        if self._kn is not None and self._kn.is_ready:
            probs, _ = self._kn.predict(x)
            p = probs[0] if probs.ndim > 1 else probs
            if p.size >= 3:
                short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])
            else:
                short_p, flat_p, long_p = 0.33, 0.34, 0.33
        else:
            short_p, flat_p, long_p = self._heuristic(snapshot)

        direction, pu, pd, pf = direction_from_probs(
            short_p, flat_p, long_p, min_confidence=self.min_confidence
        )
        conf = max(pu, pd) if direction != "flat" else pf
        return HorizonForecast(
            horizon_bars=self.horizon_bars,
            gain_threshold=self.gain_threshold,
            prob_up=pu,
            prob_down=pd,
            prob_flat=pf,
            direction=direction,
            confidence=conf,
            model_id=self.model_id,
        )

    @staticmethod
    def _heuristic(snap: StructureSnapshot) -> tuple[float, float, float]:
        t = snap.m5_trend
        a = snap.mtf_align
        score = np.clip(t + 0.3 * a, -1.0, 1.0)
        if score > 0.15:
            return 0.2, 0.25, 0.55
        if score < -0.15:
            return 0.55, 0.25, 0.2
        return 0.33, 0.34, 0.33
