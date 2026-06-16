"""ConfigValidator 基本校验。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ZhuLong.Core"))

# 轻量：直接构造等价逻辑测试（C# 校验在集成测试中由 dotnet test 覆盖）
# Python 侧仅验证 config.json 可被 json 解析且含 risk_guard


def test_config_json_has_risk_guard():
    import json

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert "risk_guard" in cfg
    rg = cfg["risk_guard"]
    assert rg["max_daily_loss_pct"] > 0
    assert rg["max_concurrent_positions"] >= 1
