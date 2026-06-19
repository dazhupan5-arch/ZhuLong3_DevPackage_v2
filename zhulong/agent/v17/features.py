"""V17 特征名与 LocationGate 输入向量。"""

from __future__ import annotations

import numpy as np

from zhulong.agent.structure_analyzer import FEATURE_NAMES
from zhulong.agent.tick_brief import StructureSnapshot

LOCATION_FEATURE_NAMES: tuple[str, ...] = (
    "pos_in_range",
    "support_dist_atr",
    "resistance_dist_atr",
    "support_strength",
    "resistance_strength",
    "mtf_trend_align",
    "atr_percentile_20d",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "direction",
    "direction_score_abs",
    "vol_regime",
    "m5_adx",
)


def extract_location_features(
    snapshot: StructureSnapshot,
    *,
    pos_in_range: float,
    direction_score: float,
    atr_percentile: float = 0.5,
) -> np.ndarray:
    vec = np.asarray(snapshot.vector, dtype=np.float32).reshape(-1)
    sup_str = float(vec[5]) if vec.size > 5 else 0.3
    res_str = float(vec[6]) if vec.size > 6 else 0.3
    hour_sin = float(vec[17]) if vec.size > 17 else 0.0
    hour_cos = float(vec[18]) if vec.size > 18 else 0.0
    dow_sin = float(vec[19]) if vec.size > 19 else 0.0
    dow_cos = float(vec[20]) if vec.size > 20 else 0.0
    m5_adx = float(vec[1]) if vec.size > 1 else 0.0
    direction = 0.0 if abs(direction_score) < 1e-9 else (1.0 if direction_score > 0 else -1.0)
    return np.array(
        [
            float(pos_in_range),
            float(snapshot.support_dist_atr),
            float(snapshot.resistance_dist_atr),
            sup_str,
            res_str,
            float(snapshot.mtf_align),
            float(atr_percentile),
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            direction,
            abs(float(direction_score)),
            float(snapshot.vol_regime),
            m5_adx,
        ],
        dtype=np.float32,
    )


def direction_score_to_forecast(
    direction_score: float,
    *,
    threshold: float = 0.35,
    model_id: str = "direction_scorer_v17",
) -> "HorizonForecast":
    from zhulong.agent.tick_brief import HorizonForecast

    score = float(direction_score)
    strength = abs(score)
    if strength < threshold:
        return HorizonForecast(
            direction="flat",
            prob_flat=1.0,
            prob_up=0.0,
            prob_down=0.0,
            confidence=0.0,
            model_id=model_id,
        )
    if score > 0:
        return HorizonForecast(
            direction="long",
            prob_up=strength,
            prob_down=max(0.0, 1.0 - strength) * 0.5,
            prob_flat=max(0.0, 1.0 - strength),
            confidence=strength,
            model_id=model_id,
        )
    return HorizonForecast(
        direction="short",
        prob_down=strength,
        prob_up=max(0.0, 1.0 - strength) * 0.5,
        prob_flat=max(0.0, 1.0 - strength),
        confidence=strength,
        model_id=model_id,
    )
