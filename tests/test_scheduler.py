"""自动调度单元测试。"""

from __future__ import annotations

import pandas as pd

from zhulong.scheduler.context import SchedulerContext
from zhulong.scheduler.market_state import SchedulerMarketState, SchedulerStateMachine
from zhulong.scheduler.risk_manager import SchedulerRiskManager
from zhulong.scheduler.scheduler_core import SchedulerCore
from zhulong.scheduler.types import ModelPrediction
from zhulong.scheduler.weight_allocator import WeightAllocator
from zhulong.strategies.base import StrategyContext


class _WinrateStub:
    def __init__(self, winrates: dict[str, float]) -> None:
        self._wr = winrates

    def is_macro_silence(self) -> bool:
        return False

    def get_atr_ratio(self, symbol: str, lookback: int = 100) -> float:
        return 1.0

    def get_adx(self, symbol: str, period: int = 14) -> float:
        return 30.0

    def get_recent_winrate(self, symbol: str, window: int = 20) -> float:
        return self._wr.get(symbol, 0.55)


def _m5_context() -> StrategyContext:
    idx = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    n = len(idx)
    m5 = pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": [1.1] * n,
            "low": [0.9] * n,
            "close": [1.0] * n,
            "volume": [1.0] * n,
        },
        index=idx,
    )
    return StrategyContext({"XAUUSD": m5, "USOIL": m5}, config={})


def test_weight_allocator_respects_target_winrate() -> None:
    wa = WeightAllocator(
        {
            "base_weights": {"XAUUSD": 0.4, "USOIL": 0.6},
            "target_winrate": {"XAUUSD": 0.55, "USOIL": 0.65},
        }
    )
    for _ in range(10):
        wa.update("XAUUSD", True)
    for _ in range(10):
        wa.update("USOIL", False)
    xau_w = wa.compute_weight("XAUUSD", 0.8)
    oil_w = wa.compute_weight("USOIL", 0.8)
    assert xau_w > oil_w


def test_weight_normalized_sums_to_one() -> None:
    wa = WeightAllocator({"base_weights": {"XAUUSD": 0.4, "USOIL": 0.6}})
    norm = wa.compute_normalized({"XAUUSD": 0.5, "USOIL": 0.5})
    assert abs(sum(norm.values()) - 1.0) < 1e-6


def test_state_machine_model_degraded() -> None:
    sm = SchedulerStateMachine({"model_degradation_winrate": 0.45, "degradation_window": 20})
    ctx = _WinrateStub({"XAUUSD": 0.40, "USOIL": 0.55})
    state = sm.update(ctx)
    assert state == SchedulerMarketState.MODEL_DEGRADED
    assert sm.get_active_strategy() == "trend_system"


def test_state_machine_volatile_on_macro_silence() -> None:
    sm = SchedulerStateMachine({})

    class _Silence(_WinrateStub):
        def is_macro_silence(self) -> bool:
            return True

    state = sm.update(_Silence({"XAUUSD": 0.6, "USOIL": 0.6}))
    assert state == SchedulerMarketState.VOLATILE
    assert sm.get_active_strategy() == "spread_hedge"


def test_risk_manager_blocks_consecutive_losses() -> None:
    rm = SchedulerRiskManager(
        {"max_consecutive_losses": 3, "max_daily_loss_r": 99, "max_total_drawdown_r": 99}
    )
    for _ in range(3):
        rm.update(-0.01)
    assert not rm.can_open_position()
    assert "连亏" in rm.block_reason


def test_risk_manager_blocks_daily_loss() -> None:
    rm = SchedulerRiskManager({"max_daily_loss_r": 0.15, "max_consecutive_losses": 99})
    rm.update(-0.16)
    assert not rm.can_open_position()


def test_scheduler_core_emits_when_models_agree() -> None:
    core = SchedulerCore(
        {
            "weight_allocator": {"base_weights": {"XAUUSD": 0.4, "USOIL": 0.6}},
            "state_machine": {"primary_symbol": "XAUUSD", "adx_threshold": 20},
            "risk_manager": {},
            "min_emit_weight": 0.01,
        }
    )
    ctx = SchedulerContext(_m5_context(), core.weight_allocator, core.risk_manager)
    preds = {
        "XAUUSD": ModelPrediction("XAUUSD", 1, 0.85, 2000, 1990, 2020),
        "USOIL": ModelPrediction("USOIL", 1, 0.90, 70, 68, 74),
    }
    outs = core.process_model_outputs(preds, ctx)
    assert len(outs) >= 1
    assert all(o.direction == "buy" for o in outs if o.direction != "flat")


def test_scheduler_core_conflict_returns_flat() -> None:
    core = SchedulerCore(
        {
            "weight_allocator": {"base_weights": {"XAUUSD": 0.5, "USOIL": 0.5}},
            "state_machine": {"primary_symbol": "XAUUSD"},
            "risk_manager": {},
        }
    )
    ctx = SchedulerContext(_m5_context(), core.weight_allocator, core.risk_manager)
    preds = {
        "XAUUSD": ModelPrediction("XAUUSD", 1, 0.9, 2000, 1990, 2020),
        "USOIL": ModelPrediction("USOIL", -1, 0.9, 70, 72, 66),
    }
    outs = core.process_model_outputs(preds, ctx)
    assert len(outs) == 1
    assert outs[0].direction == "flat"
    assert outs[0].reject_reason == "direction_conflict"
