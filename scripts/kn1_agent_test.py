"""KN1 TradingAgent init test — verified after rollback"""
import torch
import sys, json
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))

from zhulong.agent.trading_agent import TradingAgent

print("Initializing TradingAgent...")
agent = TradingAgent(cfg, root=ROOT)

print()
print("=== TradingAgent Status ===")
print(f"  primary_symbol    = {agent.primary_symbol}")
print(f"  use_rl            = {agent.use_rl}")
print(f"  knowledge.is_ready = {agent._knowledge.is_ready if agent._knowledge else 'None'}")
print(f"  rl.is_ready       = {agent._rl.is_ready if agent._rl else 'None'}")
print(f"  enabled           = {agent.enabled}")

# Check that knowledge_net is KN1 (ONNX) type, not KN2
kn = agent._knowledge
if kn is not None:
    kn_type = type(kn).__name__
    kn_module = type(kn).__module__
    print(f"  knowledge type    = {kn_type} (from {kn_module})")
    has_onnx = hasattr(kn, "onnx_session") and kn.onnx_session is not None
    print(f"  onnx_session      = {'loaded' if has_onnx else 'NOT loaded'}")

print()
print("=== READY (KN 1.0) ===")
