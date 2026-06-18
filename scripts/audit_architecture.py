"""综合架构检查：KN2 V16 + 黄金模型"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("ARCHITECTURE AUDIT: KN2 Integration + Gold Model")
print("=" * 60)

checks = []

print("\n--- 1. Config Check ---")
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
kn2 = cfg.get("kn2", {})
print(f"  kn2.enabled:       {kn2.get('enabled')}")
print(f"  kn2.shadow_mode:   {kn2.get('shadow_mode')}")
print(f"  kn2.model_path:    {kn2.get('model_path')}")
print(f"  kn2.min_confidence:{kn2.get('min_confidence')}")
print(f"  agent top enabled: {cfg.get('enabled')}")
checks.append(("config", True))

print("\n--- 2. Causal Graph ---")
from zhulong.agent.causal_inference import load_causal_graph

graph = load_causal_graph()
has_gold = "XAUUSD" in graph.get("symbols", {})
has_oil = "USOIL" in graph.get("symbols", {})
print(f"  JSON fallback: {has_gold} / {has_oil}")
print(f"  Symbols: {list(graph.get('symbols', {}).keys())}")
checks.append(("causal_graph", has_gold and has_oil))

print("\n--- 3. KN2 Module Imports ---")
try:
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

    print("  KN2Inference + encode_position_state: OK")
    checks.append(("kn2_imports", True))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("kn2_imports", False))

print("\n--- 4. Position State Encoder ---")
try:
    ps0 = encode_position_state()
    print(f"  No position: {ps0}")
    ps1 = encode_position_state(direction=1.0)
    print(f"  Long sign: {ps1}")
    checks.append(("pos_state", True))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("pos_state", False))

print("\n--- 5. TradingAgent tick_symbols ---")
try:
    from zhulong.agent.trading_agent import TradingAgent

    src = (ROOT / "zhulong/agent/trading_agent.py").read_text(encoding="utf-8")
    has_multi = "for sym in ordered" in src
    has_pos_dup = "_has_open_position" in src
    print(f"  multi-symbol tick: {has_multi}")
    print(f"  position duplicate bypass: {has_pos_dup}")
    checks.append(("trading_agent", has_multi and has_pos_dup))
except Exception as e:
    print(f"  FAIL: {e}")
    checks.append(("trading_agent", False))

print("\n--- 6. Model Files ---")
for name, rel in [
    ("Gold V16", "models/kn2_trader_v16.pth"),
    ("Horizon", "models/horizon_v16.onnx"),
]:
    pth = ROOT / rel
    print(f"  {name}: {pth.exists()} ({pth.stat().st_size if pth.exists() else 0} B)")

print(f"\n{'=' * 60}")
passed = sum(1 for _, ok in checks if ok)
total = len(checks)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
print(f"  Result: {passed}/{total}")
print("=" * 60)
sys.exit(0 if passed == total else 1)
