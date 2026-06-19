"""V17 DirectionScorer 推理：LightGBM 回归 → direction_score ∈ [-1, +1]。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from zhulong.agent.rl_agent import _resolve_model_path
from zhulong.agent.tick_brief import StructureSnapshot

logger = logging.getLogger(__name__)


class DirectionScorer:
    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        arch = config.get("architecture") or {}
        ds_cfg = arch.get("direction_scorer") or {}
        self.min_abs_score = float(ds_cfg.get("min_abs_score", 0.35))
        self.model_path = _resolve_model_path(
            str(ds_cfg.get("model_path", "models/direction_scorer/direction_scorer.lgb")),
            root,
        )
        meta_path = self.model_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            alt = self.model_path.parent / "meta.json"
            meta_path = alt if alt.is_file() else meta_path
        self.meta: dict = {}
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        keep = self.meta.get("keep_features")
        self.keep_features: list[int] | None = list(keep) if keep else None
        self._model = None
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.load_error = f"model_missing:{self.model_path}"
            return
        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(self.model_path))
            logger.info("DirectionScorer 已加载: %s", self.model_path)
        except Exception as ex:
            self.load_error = f"{type(ex).__name__}:{ex}"
            logger.warning("DirectionScorer 加载失败: %s", ex)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _prepare_x(self, snapshot: StructureSnapshot) -> np.ndarray:
        x = np.asarray(snapshot.vector, dtype=np.float32).reshape(1, -1)
        if self.keep_features:
            x = x[:, self.keep_features]
        return x

    def predict(self, snapshot: StructureSnapshot) -> float:
        if not self.is_ready:
            return 0.0
        x = self._prepare_x(snapshot)
        score = float(self._model.predict(x)[0])
        return max(-1.0, min(1.0, score))

    def predict_batch(self, struct: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return np.zeros(len(struct), dtype=np.float32)
        x = np.asarray(struct, dtype=np.float32)
        if self.keep_features:
            x = x[:, self.keep_features]
        scores = self._model.predict(x)
        return np.clip(scores, -1.0, 1.0).astype(np.float32)
