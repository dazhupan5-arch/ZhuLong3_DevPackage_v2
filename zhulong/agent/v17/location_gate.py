"""V17 LocationGate 推理：XGBoost 二分类 → location_quality ∈ [0, 1]。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from zhulong.agent.rl_agent import _resolve_model_path
from zhulong.agent.tick_brief import StructureSnapshot
from zhulong.agent.v17.features import LOCATION_FEATURE_NAMES, extract_location_features

logger = logging.getLogger(__name__)


class LocationGate:
    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        arch = config.get("architecture") or {}
        lg_cfg = arch.get("location_gate") or {}
        self.min_quality = float(lg_cfg.get("min_quality", 0.60))
        self.required = bool(lg_cfg.get("required", True))
        self.model_path = _resolve_model_path(
            str(lg_cfg.get("model_path", "models/location_gate/location_gate.xgb")),
            root,
        )
        meta_path = self.model_path.parent / "meta.json"
        self.meta: dict = {}
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        self._model = None
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.load_error = f"model_missing:{self.model_path}"
            return
        try:
            import xgboost as xgb

            self._model = xgb.XGBClassifier()
            self._model.load_model(str(self.model_path))
            logger.info("LocationGate 已加载: %s", self.model_path)
        except Exception as ex:
            self.load_error = f"{type(ex).__name__}:{ex}"
            logger.warning("LocationGate 加载失败: %s", ex)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(
        self,
        snapshot: StructureSnapshot,
        *,
        pos_in_range: float,
        direction_score: float,
        atr_percentile: float = 0.5,
    ) -> float:
        if not self.is_ready:
            return 0.0 if self.required else 1.0
        if abs(direction_score) < 1e-9:
            return 0.0
        x = extract_location_features(
            snapshot,
            pos_in_range=pos_in_range,
            direction_score=direction_score,
            atr_percentile=atr_percentile,
        ).reshape(1, -1)
        prob = float(self._model.predict_proba(x)[0, 1])
        return max(0.0, min(1.0, prob))

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return np.zeros(len(features), dtype=np.float32)
        has_dir = np.abs(features[:, 11]) > 1e-9
        out = np.zeros(len(features), dtype=np.float32)
        if not has_dir.any():
            return out
        probs = self._model.predict_proba(features[has_dir])[:, 1]
        out[has_dir] = probs
        return np.clip(out, 0.0, 1.0)
