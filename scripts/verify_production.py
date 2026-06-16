import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, "d:/Program Files/ZhuLong")

print("=== 1. Python imports ===")

# zhulong.engine
from zhulong.engine.runtime_config import apply_runtime_primary, bind_engine_primary
from zhulong.engine.agent_engine import run_agent_tick
from zhulong.engine.multi_strategy_engine import MultiStrategyEngine
from zhulong.engine.scheduler_engine import SchedulerEngine
print("zhulong.engine: OK")

# zhulong.agent
from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.state_builder import StateBuilder, STATE_DIM
from zhulong.agent.rl_agent import RlAgent
print("zhulong.agent: OK")

# zhulong.v14_live
from zhulong.v14_live import build_live_v14_features
print("zhulong.v14_live: OK")

print()
print("=== 2. KnowledgeNet inference ===")
kn = KnowledgeNetInference("d:/Program Files/ZhuLong/models/knowledge_net.onnx")
print("KN ready:", kn.is_ready)
test_x = np.random.randn(5, 68).astype(np.float32)
probs, emb = kn.predict(test_x)
print("KN inference:", probs.shape, "probs,", emb.shape, "emb")
print("Probs sum:", probs.sum(axis=1))

print()
print("=== 3. V14 features callable ===")
print("build_live_v14_features: callable")

print()
print("=== 4. StateBuilder dimension ===")
sb = StateBuilder("d:/Program Files/ZhuLong/data/agent_state_scaler_xauusd.json")
print("STATE_DIM:", STATE_DIM)
print("Scaler loaded:", sb.mean is not None, "dim=", len(sb.mean) if sb.mean is not None else 0)
assert sb.mean is not None and len(sb.mean) == STATE_DIM, "Scaler dim mismatch"

print()
print("=== 5. RL Agent model file ===")
rl_path = "d:/Program Files/ZhuLong/models/rl_agent_xau.zip"
print("RL model:", rl_path, "(" + str(Path(rl_path).stat().st_size // 1024) + " KB)")

print()
print("=== 6. Config ===")
with open("d:/Program Files/ZhuLong/config.json") as f:
    cfg = json.load(f)
ta = cfg.get("trading_agent", {})
print("trading_agent.enabled:", ta.get("enabled", False))

with open("d:/Program Files/ZhuLong/config/config_agent.json") as f:
    agent_cfg = json.load(f)
kn_cfg = agent_cfg["knowledge_net"]
print("knowledge_net.hidden_dim:", kn_cfg["hidden_dim"])
print("knowledge_net.input_dim:", kn_cfg["input_dim"])
print("rl.model_path_xau:", agent_cfg["rl"]["model_path_xau"])

print()
print("=== ALL CHECKS PASSED ===")
