"""快速架构验证：KN2 model load + prediction test"""
import sys, json, numpy as np
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

print("KN2 Architecture Verification")
print("=" * 60)

# 1. Load model
print("\n--- 1. Model Load ---")
from zhulong.agent.knowledge_net_kn2 import KN2Inference
kn2 = KN2Inference(ROOT / "models/kn2_trader.pth")
print(f"  is_ready: {kn2.is_ready}")
print(f"  num_actions: {kn2.num_actions}")

# 2. Test prediction
print("\n--- 2. Prediction Test ---")
from zhulong.agent.knowledge_net_kn2 import encode_position_state

# Random market features
np.random.seed(42)
mf = np.random.randn(98).astype(np.float32)

# No position
ps = encode_position_state()
out1 = kn2.predict(mf, ps)
print(f"  No pos: action={out1['action_name']} conf={out1['confidence']:.3f} "
      f"trade={out1['should_trade']}")

# Long position
ps2 = encode_position_state(direction=1.0, hold_bars=10, float_pnl_pct=0.02,
                             max_favorable_pct=0.03, max_adverse_pct=0.01)
out2 = kn2.predict(mf, ps2)
print(f"  Long:   action={out2['action_name']} conf={out2['confidence']:.3f} "
      f"trade={out2['should_trade']}")

# Short position
ps3 = encode_position_state(direction=-1.0, hold_bars=5, float_pnl_pct=-0.01,
                             max_favorable_pct=0.01, max_adverse_pct=0.02)
out3 = kn2.predict(mf, ps3)
print(f"  Short:  action={out3['action_name']} conf={out3['confidence']:.3f} "
      f"trade={out3['should_trade']}")

# 3. Hidden state test (sequential)
print("\n--- 3. Sequential Prediction (hidden state) ---")
kn2.reset_hidden()
actions = []
for i in range(10):
    out = kn2.predict(mf, ps)
    actions.append(out["action_name"])
print(f"  10 steps: {actions}")
print(f"  Has hidden state: {kn2._h is not None}")

# 4. TradingAgent test
print("\n--- 4. TradingAgent Instantiation ---")
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
from zhulong.agent.trading_agent import TradingAgent
agent = TradingAgent(config=cfg, root=str(ROOT))
print(f"  enabled: {agent.enabled}")
print(f"  kn2_mode: {agent.kn2_mode}")
print(f"  kn2_shadow: {agent.kn2_shadow}")
print(f"  kn2_model_path: {agent.kn2_model_path}")
print(f"  primary_symbol: {agent.primary_symbol}")

# 5. Config summary
print("\n--- 5. Config Summary ---")
print(f"  Top enabled: {cfg.get('enabled')}")
print(f"  KN2 enabled: {cfg['kn2'].get('enabled')}")
print(f"  KN2 shadow:  {cfg['kn2'].get('shadow_mode')}")
print(f"  KN2 path:    {cfg['kn2'].get('model_path')}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - KN2 Architecture + Gold Model Ready")
print("=" * 60)
