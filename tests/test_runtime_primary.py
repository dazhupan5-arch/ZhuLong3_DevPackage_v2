"""运行时主品种覆盖测试。"""

from zhulong.engine.runtime_config import apply_runtime_primary, bind_engine_primary
from zhulong.engine.multi_strategy_engine import MultiStrategyEngine


def test_apply_runtime_primary_overrides_config() -> None:
    cfg = {"state_machine": {"primary_symbol": "XAUUSD"}, "scheduler_core": {"state_machine": {}}}
    sym = apply_runtime_primary(cfg, "USOIL")
    assert sym == "USOIL"
    assert cfg["state_machine"]["primary_symbol"] == "USOIL"
    assert cfg["scheduler_core"]["state_machine"]["primary_symbol"] == "USOIL"


def test_bind_engine_primary_multi_strategy() -> None:
    engine = MultiStrategyEngine({"state_machine": {"primary_symbol": "XAUUSD"}, "symbols": {}})
    bind_engine_primary(engine, "USOIL")
    assert engine.primary_symbol == "USOIL"
