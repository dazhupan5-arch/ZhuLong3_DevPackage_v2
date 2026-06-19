"""V17 tick 路径：Structure → DirectionScorer → LocationGate → ExecutionComposerV17。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from zhulong.agent.kn2_location_labels import compute_pos_in_range
from zhulong.agent.structure_service import StructureService
from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast
from zhulong.agent.v17.direction_scorer import DirectionScorer
from zhulong.agent.v17.execution_composer_v17 import ExecutionComposerV17
from zhulong.agent.v17.features import direction_score_to_forecast
from zhulong.agent.v17.location_gate import LocationGate

logger = logging.getLogger(__name__)


class V17Pipeline:
    def __init__(self, root, config: dict[str, Any]) -> None:
        self.root = root
        self.config = config
        self.structure_service = StructureService(config.get("structure_analyzer"))
        self.direction_scorer = DirectionScorer(root, config)
        self.location_gate = LocationGate(root, config)
        self.composer = ExecutionComposerV17(config)
        ds_cfg = (config.get("architecture") or {}).get("direction_scorer") or {}
        self.min_abs_score = float(ds_cfg.get("min_abs_score", 0.35))

    @property
    def is_ready(self) -> bool:
        return self.direction_scorer.is_ready and self.location_gate.is_ready

    def run(
        self,
        m5: pd.DataFrame,
        loc: int,
        *,
        close: float,
        atr: float,
        consecutive_losses: int = 0,
        causal_score: float = 0.0,
    ) -> tuple[HorizonForecast, ExecutionPlan, dict[str, Any]]:
        snapshot = self.structure_service.snapshot_from_row(m5, loc)
        closes = m5["close"].to_numpy(dtype=np.float32)
        pos_arr = compute_pos_in_range(closes, window=48)
        pos_in_range = float(pos_arr[loc]) if loc < len(pos_arr) else 0.5
        direction_score = self.direction_scorer.predict(snapshot)
        location_quality = self.location_gate.predict(
            snapshot,
            pos_in_range=pos_in_range,
            direction_score=direction_score,
        )
        forecast = direction_score_to_forecast(
            direction_score,
            threshold=self.min_abs_score,
            model_id=str(
                (self.config.get("architecture") or {})
                .get("direction_scorer", {})
                .get("model_id", "direction_scorer_v17")
            ),
        )
        regime = str(snapshot.zigzag_phase or "")
        plan = self.composer.compose_v17(
            direction_score=direction_score,
            location_quality=location_quality,
            snapshot=snapshot,
            close=close,
            atr=atr,
            pos_in_range=pos_in_range,
            consecutive_losses=consecutive_losses,
            regime=regime,
            causal_score=causal_score,
            location_gate_required=self.location_gate.required,
        )
        meta = {
            "direction_score": round(direction_score, 4),
            "location_quality": round(location_quality, 4),
            "struct": np.asarray(snapshot.vector, dtype=np.float32),
        }
        return forecast, plan, meta
