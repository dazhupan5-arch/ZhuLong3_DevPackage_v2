"""KN1 Rollback Audit — verify system runs on KN 1.0 after rollback"""
import sys, json, numpy as np
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

print("=" * 55)
print("KN1 ROLLBACK AUDIT")
print("=" * 55)

all_pass = True

# ── 1. Config ──
print("\n[1] Config:")
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
kn2_enabled = cfg["kn2"]["enabled"]
kn1_path = cfg["knowledge_net"]["model_path"]
rl_path = cfg["rl"]["model_path_xau"]
use_rl = cfg.get("use_rl", False)
print(f"    kn2.enabled = {kn2_enabled}")
print(f"    knowledge_net model_path = {kn1_path}")
print(f"    rl model_path_xau = {rl_path}")
print(f"    use_rl = {use_rl}")
if kn2_enabled:
    print("    [WARN] kn2 is still enabled!")
    all_pass = False

# ── 2. Model file existence ──
print("\n[2] Model Files:")
files = [
    ("models/knowledge_net.onnx", "KN1 ONNX model"),
    ("models/knowledge_scaler.pkl", "KN1 scaler"),
    ("models/knowledge_net.pth", "KN1 PyTorch model"),
    ("models/rl_agent_xau/actor.pth", "RL actor"),
    ("models/rl_agent_xau/critic.pth", "RL critic"),
]
for fpath, desc in files:
    p = ROOT / fpath
    ok = p.exists()
    sz = p.stat().st_size if ok else 0
    marker = "[OK]" if ok else "[FAIL]"
    print(f"    {marker} {fpath:40s} {sz:>10,} bytes  ({desc})")
    if not ok:
        all_pass = False

# ── 3. TradingAgent import ──
print("\n[3] TradingAgent:")
try:
    from zhulong.agent.trading_agent import TradingAgent
    # Check that the right knowledge_net is imported
    import zhulong.agent.trading_agent as ta_mod
    source = ta_mod.__file__
    print(f"    trading_agent.py source = {source}")
    print(f"    import OK")
except Exception as e:
    print(f"    [FAIL] import: {e}")
    all_pass = False

# ── 4. verify it imports knowledge_net NOT knowledge_net_kn2 ──
print("\n[4] Import Path Check:")
import zhulong.agent.trading_agent as ta
import inspect
kn_imports = [line for line in inspect.getsource(ta).split("\n") 
              if "knowledge_net" in line.lower() and ("import" in line.lower() or "from" in line.lower())]
for line in kn_imports:
    print(f"    {line.strip()}")
has_kn1 = any("from zhulong.agent.knowledge_net import" in l for l in kn_imports)
has_kn2 = any("knowledge_net_kn2" in l for l in kn_imports)
if has_kn1 and not has_kn2:
    print("    [OK] Only knowledge_net (KN1) imported")
elif has_kn2:
    print("    [WARN] knowledge_net_kn2 still imported!")
else:
    print("    [WARN] No knowledge_net imports found")

# ── 5. Live TradingAgent init test ──
print("\n[5] TradingAgent Initialization:")
try:
    agent = TradingAgent((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"), root=ROOT)
    print(f"    kn2_mode = {agent.kn2_mode}")
    print(f"    _knowledge loaded = {agent._knowledge is not None}")
    print(f"    _rl loaded = {agent._rl is not None}")
    if agent.kn2_mode:
        print("    [WARN] agent.kn2_mode is True")
        all_pass = False
    else:
        print("    [OK] kn2_mode is False — using KN1 path")
    if agent._knowledge is not None:
        print(f"    KN1 is_ready = {agent._knowledge.is_ready}")
        if not agent._knowledge.is_ready:
            print("    [WARN] KN1 not ready!")
            all_pass = False
    else:
        print("    [WARN] _knowledge is None — KN1 not loaded")
        all_pass = False
except Exception as e:
    print(f"    [FAIL] Agent init: {e}")
    all_pass = False

# ── FINAL ──
print("\n" + "=" * 55)
if all_pass:
    print("RESULT: ALL CHECKS PASSED")
else:
    print("RESULT: SOME CHECKS FAILED — see [FAIL]/[WARN] above")
print("=" * 55)
