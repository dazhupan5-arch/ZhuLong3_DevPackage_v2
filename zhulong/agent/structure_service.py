"""L1 结构层：StructureAnalyzer 唯一出口。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from zhulong.agent.structure_analyzer import FEATURE_DIM, StructureAnalyzer
from zhulong.agent.tick_brief import StructureSnapshot


class StructureService:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._sa = StructureAnalyzer(config or {})

    def snapshot_from_row(self, m5: pd.DataFrame, idx: int) -> StructureSnapshot:
        row = self._sa.compute_row(m5, idx, mtf=self._sa._build_mtf_context(m5))
        return self._row_to_snapshot(row)

    def snapshot_latest(self, m5: pd.DataFrame) -> StructureSnapshot:
        if m5.empty:
            return StructureSnapshot(vector=[0.0] * FEATURE_DIM)
        return self.snapshot_from_row(m5.sort_index(), len(m5) - 1)

    @staticmethod
    def _row_to_snapshot(row: np.ndarray) -> StructureSnapshot:
        r = np.asarray(row, dtype=np.float32).reshape(-1)
        if r.size < FEATURE_DIM:
            r = np.pad(r, (0, FEATURE_DIM - r.size))
        trend = float(r[0])
        mtf = float(r[26]) if r.size > 26 else 0.0
        if trend > 0.05 and mtf >= 0:
            phase = "up_swing"
        elif trend < -0.05 and mtf <= 0:
            phase = "down_swing"
        else:
            phase = "range"
        return StructureSnapshot(
            vector=[float(x) for x in r[:FEATURE_DIM]],
            m5_trend=trend,
            support_dist_atr=float(r[3]) if r.size > 3 else 0.0,
            resistance_dist_atr=float(r[4]) if r.size > 4 else 0.0,
            mtf_align=mtf,
            vol_regime=float(r[16]) if r.size > 16 else 1.0,
            zigzag_phase=phase,
        )
