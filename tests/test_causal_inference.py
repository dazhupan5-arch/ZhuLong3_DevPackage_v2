"""因果推理模块测试。"""

from __future__ import annotations

import numpy as np

from zhulong.agent.causal_inference import (
    CausalInference,
    CounterfactualPredictor,
    fuse_knowledge_with_causal,
)


def test_fuse_knowledge_causal_bullish():
    probs = np.array([[0.4, 0.3, 0.3]], dtype=np.float32)
    fused = fuse_knowledge_with_causal(probs, causal_pred=0.5, weight_causal=0.3)
    assert fused.shape == (1, 3)
    assert abs(fused.sum() - 1.0) < 1e-5
    assert fused[0, 2] > probs[0, 2]


def test_causal_inference_heuristic_without_coef():
    ci = CausalInference("models/missing_causal.pkl", symbol="XAUUSD")
    assert not ci.is_ready
    pred = ci.predict_price_change(1.0)
    assert isinstance(pred, float)


def test_counterfactual_luck_zero_without_causal():
    cf = CounterfactualPredictor(None)
    luck = cf.luck_pnl_r(
        direction=1.0,
        entry_price=2000.0,
        position_frac=1.0,
        initial_balance=10000.0,
        exogenous_sum=0.5,
        hold_bars=6,
    )
    assert luck == 0.0


def test_macro_shock_from_struct():
    ci = CausalInference("models/missing_causal.pkl")
    shock = ci.macro_shock_from_bar(np.array([0.2, 0.1, -0.05], dtype=np.float32))
    assert -3.0 <= shock <= 3.0
