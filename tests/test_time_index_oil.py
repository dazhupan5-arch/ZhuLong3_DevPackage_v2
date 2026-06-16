"""时间索引归一化与 USOIL 实机路径回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.scheduler.context import SchedulerContext
from zhulong.scheduler.scheduler_core import SchedulerCore
from zhulong.scheduler.types import ModelPrediction
from zhulong.strategies.base import StrategyContext
from zhulong.training.oil_v1.backtest import _eia_blackout_mask, h1_extreme_trend_filter
from zhulong.utils.time_index import normalize_m5_index


def _m5_tz_aware(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-06-08 10:00", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 70.0 + rng.normal(0, 0.2, n).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )


def test_eia_blackout_accepts_tz_aware_index() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-04 03:00", tz="UTC")])
    mask = _eia_blackout_mask(idx)
    assert mask.shape == (1,)
    assert isinstance(mask[0], (bool, np.bool_))


def test_h1_extreme_filter_accepts_tz_aware_bars() -> None:
    m5 = _m5_tz_aware()
    t = pd.DatetimeIndex([m5.index[-1]])
    out = h1_extreme_trend_filter(m5, t, np.array([1]))
    assert out.shape == (1,)


def test_bars_to_df_strips_timezone() -> None:
    import sys
    from pathlib import Path

    engine_dir = Path(__file__).resolve().parent.parent / "ZhuLong.PythonEngine"
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    from inference_cli import _bars_to_df

    ts = int(pd.Timestamp("2026-06-08 12:00", tz="UTC").timestamp())
    df = _bars_to_df([[ts, 1.0, 1.1, 0.9, 1.0, 10.0]])
    assert df.index.tz is None


def test_scheduler_core_all_hold_returns_reason() -> None:
    idx = pd.date_range("2026-01-01", periods=50, freq="5min")
    m5 = pd.DataFrame(
        {"open": [1.0] * 50, "high": [1.1] * 50, "low": [0.9] * 50, "close": [1.0] * 50, "volume": [1.0] * 50},
        index=idx,
    )
    ctx = SchedulerContext(
        StrategyContext({"XAUUSD": m5}, config={}),
        SchedulerCore({"weight_allocator": {}, "state_machine": {}, "risk_manager": {}}).weight_allocator,
        SchedulerCore({"weight_allocator": {}, "state_machine": {}, "risk_manager": {}}).risk_manager,
    )
    core = SchedulerCore(
        {
            "weight_allocator": {"base_weights": {"XAUUSD": 1.0}},
            "state_machine": {"primary_symbol": "XAUUSD"},
            "risk_manager": {},
        }
    )
    preds = {
        "XAUUSD": ModelPrediction("XAUUSD", 0, 0.55, 1.0, 0.0, 0.0, reject_reason="no_signal"),
    }
    outs = core.process_model_outputs(preds, ctx)
    assert len(outs) == 1
    assert outs[0].direction == "flat"
    assert "XAUUSD:no_signal" in outs[0].reject_reason


def test_normalize_m5_index() -> None:
    df = normalize_m5_index(_m5_tz_aware(10))
    assert df.index.tz is None
