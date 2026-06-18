"""归因引擎测试。"""

from __future__ import annotations

from zhulong.attribution.engine import AttributionEngine


def test_attribution_horizon_bins():
    rows = [
        {
            "pnl_percent": 1.2,
            "attribution_json": '{"horizon_direction":"long","cognition_regime":"trend","filter_reason":"","kn2_should_trade":true,"horizon_confidence":0.72}',
        },
        {
            "pnl_percent": -0.8,
            "attribution_json": '{"horizon_direction":"long","cognition_regime":"ranging","filter_reason":"location_gate","kn2_should_trade":false,"horizon_confidence":0.61}',
        },
        {
            "pnl_percent": 0.5,
            "attribution_json": '{"horizon_direction":"short","cognition_regime":"trend","filter_reason":"","kn2_should_trade":true,"horizon_confidence":0.68}',
        },
    ]
    report = AttributionEngine(min_samples=2).analyze(rows)
    assert report.total_trades == 3
    assert len(report.horizon_bins) >= 2
    assert any(s.key == "meta_learning" or "location" in s.key for s in report.tune_suggestions) or report.win_rate > 0
