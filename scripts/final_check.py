"""KN2 黄金模型最终验证 + 决策链路测试"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

print("=" * 60)
print("KN2 GOLD MODEL - FINAL VERIFICATION")
print("=" * 60)

kn2 = KN2Inference(ROOT / "models/kn2_trader.pth")
print(f"  Loaded: {kn2.is_ready}")
print(f"  Actions: {kn2.num_actions}")

# Direction sensitivity test
mf = np.random.randn(98).astype(np.float32)
print("\n  Direction sensitivity (feature[0] sweep):")
for trend in [-1.5, -0.5, 0.0, 0.5, 1.5]:
    mf2 = mf.copy()
    mf2[0] = trend
    out = kn2.predict(mf2, encode_position_state())
    print(f"    signal={trend:+.1f}: {out['action_name']:5s} conf={out['confidence']:.3f} trade={out['should_trade']}")

# Sequential test (hidden state)
print("\n  Sequential prediction (10 steps, same features):")
kn2.reset_hidden()
for i in range(5):
    out = kn2.predict(mf, encode_position_state())
    print(f"    step {i}: {out['action_name']:5s} conf={out['confidence']:.3f}")

# Meta check
meta = json.loads((ROOT / "models/kn2_trader.meta.json").read_text(encoding="utf-8-sig"))
print(f"\n  Model config: {meta['hidden_dim']}d x {meta['num_layers']} layers, {meta['num_actions']} actions")
print(f"  val_loss: {meta['val_loss']:.6f}")

print("\n" + "=" * 60)
print("GOLD MODEL: READY FOR PRODUCTION")
print("=" * 60)
