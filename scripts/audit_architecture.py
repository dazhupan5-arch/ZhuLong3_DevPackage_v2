"""综合架构检查：KN2 + 黄金模型"""
import sys, json, numpy as np
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("ARCHITECTURE AUDIT: KN2 Integration + Gold Model")
print("=" * 60)

checks = []

# 1. Config
print("\n--- 1. Config Check ---")
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
kn2 = cfg.get("kn2", {})
print(f"  kn2.enabled:       {kn2.get('enabled')}")
print(f"  kn2.shadow_mode:   {kn2.get('shadow_mode')}")
print(f"  kn2.model_path:    {kn2.get('model_path')}")
print(f"  kn2.min_confidence:{kn2.get('min_confidence')}")
print(f"  agent top enabled: {cfg.get('enabled')}")
checks.append(("config", True))

# 2. Causal graph
print("\n--- 2. Causal Graph ---")
from zhulong.agent.causal_inference import load_causal_graph
graph = load_causal_graph()
has_gold = "XAUUSD" in graph.get("symbols", {})
has_oil = "USOIL" in graph.get("symbols", {})
print(f"  JSON fallback: {has_gold} / {has_oil}")
print(f"  Symbols: {list(graph.get('symbols', {}).keys())}")
checks.append(("causal_graph", has_gold and has_oil))

# 3. KN2 module imports
print("\n--- 3. KN2 Module Imports ---")
try:
    from zhulong.agent.knowledge_net_kn2 import (
        TraderKnowledgeGRU, KN2Inference,
        _build_trader_gru_class, encode_position_state,
        train_kn2_fast,
    )
    print("  All imports: OK")
    checks.append(("kn2_imports", True))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("kn2_imports", False))

# 4. Model architecture
print("\n--- 4. Model Architecture ---")
try:
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=128, num_layers=2, embed_dim=64, num_actions=3
    )
    m = KnCls()
    params = sum(p.numel() for p in m.parameters())
    print(f"  Params: {params:,}")
    has_attrs = all(hasattr(m, a) for a in [
        "market_encoder", "pos_encoder", "gru", "action_head",
        "size_head", "sl_head", "tp_head", "conf_head", "trade_head", "embed_head"
    ])
    print(f"  All components: {has_attrs}")
    checks.append(("model_arch", has_attrs))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("model_arch", False))

# 5. Position state
print("\n--- 5. Position State Encoder ---")
try:
    ps0 = encode_position_state()
    print(f"  No position: {ps0}")
    ps1 = encode_position_state(
        direction=1.0, hold_bars=10, float_pnl_pct=0.02,
        max_favorable_pct=0.03, max_adverse_pct=0.01,
    )
    print(f"  Long: {ps1}")
    checks.append(("pos_state", True))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("pos_state", False))

# 6. TradingAgent instantiation
print("\n--- 6. TradingAgent ---")
try:
    from zhulong.agent.trading_agent import TradingAgent
    agent = TradingAgent(config=cfg, root=str(ROOT), symbol="XAUUSD")
    print(f"  kn2_mode: {agent.kn2_mode}")
    print(f"  kn2_shadow: {agent.kn2_shadow}")
    print(f"  kn2_model_path: {agent.kn2_model_path}")
    print(f"  knowledge attr: {agent.knowledge}")  # KN1
    print(f"  _kn2 attr: {agent._kn2}")
    checks.append(("trading_agent", True))
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()
    checks.append(("trading_agent", False))

# 7. Model files
print("\n--- 7. Model Files ---")
for name, pth in [("Gold", ROOT / "models/kn2_trader.pth"),
                   ("Oil", ROOT / "models/kn2_trader_oil.pth")]:
    meta = pth.with_suffix(".meta.json")
    print(f"  {name}: .pth={pth.exists()} .meta={meta.exists()}")
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8-sig"))
        print(f"    meta: {m.get('hidden_dim')}dx{m.get('num_layers')} "
              f"actions={m.get('num_actions')} val_loss={m.get('val_loss',0):.6f}")
    if pth.exists():
        print(f"    size: {pth.stat().st_size:,} bytes")

# 8. Summary
print(f"\n{'='*60}")
passed = sum(1 for _, ok in checks if ok)
total = len(checks)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
print(f"  Result: {passed}/{total}")
print("=" * 60)
